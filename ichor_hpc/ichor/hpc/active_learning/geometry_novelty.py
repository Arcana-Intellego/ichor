"""Geometry novelty scaling for Phase-B anti-overlap.

The Phase-B de-duplication metric is an aligned RMSD-like distance in
Angstrom. A single absolute threshold is awkward across differently sized or
flexible systems, so this module estimates a per-iteration motion scale and
lets Phase B express its threshold as a dimensionless multiple of that scale.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ichor.core.adversarial.geometry import aligned_mass_weighted_rmsd
from ichor.core.atoms import Atoms

from .daemon.state import DEFAULT_STATE_FILENAME, atomic_write_json, read_state
from .geometry_protocol import (
    FULLSPACE_RMSD_SCALE_MULTIPLIER,
    GEOMETRY_NOVELTY_SCORE_TRANSFORM,
    MOVEMENT_BAND_HARD_MAX_FRACTION,
    MOVEMENT_BAND_HARD_MIN_FRACTION,
    MOVEMENT_BAND_TARGET_HIGH_FRACTION,
    MOVEMENT_BAND_TARGET_LOW_FRACTION,
    MOVEMENT_BAND_TARGET_PEAK_FRACTION,
    MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION,
    MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION,
    PHASE_B_MIN_SEPARATION_SCALE,
    movement_band_fractions,
)


GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION = 1
GEOMETRY_NOVELTY_SCALE_FILENAME = "GEOMETRY_NOVELTY_SCALE.json"
EXACT_DUPLICATE_EPSILON_ANGSTROM = 1.0e-12


__all__ = [
    "EXACT_DUPLICATE_EPSILON_ANGSTROM",
    "GEOMETRY_NOVELTY_SCALE_FILENAME",
    "GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION",
    "compute_geometry_novelty_scale",
    "effective_phase_b_min_separation",
    "ensure_geometry_novelty_scale",
    "geometry_novelty_scale_path",
    "novelty_score",
    "read_geometry_novelty_scale",
    "resolve_geometry_novelty_consumers",
    "apply_geometry_novelty_to_acquisition_config",
    "scaled_distances",
    "write_geometry_novelty_scale",
]


def geometry_novelty_scale_path(iter_dir: Union[str, Path]) -> Path:
    return Path(iter_dir) / GEOMETRY_NOVELTY_SCALE_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_positive(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _percentile(values: Sequence[float], statistic: str) -> float:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute a statistic for an empty value set")
    if statistic == "p25":
        return float(np.percentile(arr, 25.0))
    if statistic == "p75":
        return float(np.percentile(arr, 75.0))
    return float(np.percentile(arr, 50.0))


def _summary(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if _finite_positive(v) is not None]
    if not clean:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    return {
        "n": int(len(clean)),
        "min": float(min(clean)),
        "p25": _percentile(clean, "p25"),
        "median": _percentile(clean, "median"),
        "p75": _percentile(clean, "p75"),
        "max": float(max(clean)),
    }


def _load_seed_payload(iter_dir: Path) -> Dict[str, Any]:
    path = iter_dir / "seeds_picked.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _load_raw_seed_records(iter_dir: Path) -> List[Dict[str, Any]]:
    data = _load_seed_payload(iter_dir)
    raw = data.get("seed_records")
    if isinstance(raw, list):
        return [dict(rec) for rec in raw if isinstance(rec, dict)]
    frame_ids = data.get("frame_ids")
    if not isinstance(frame_ids, list):
        return []
    return [
        {"seed_index": int(i), "frame_id": _safe_int(frame_id)}
        for i, frame_id in enumerate(frame_ids)
    ]


def _manifest_neighbour_ids_for_seed(
    record: Dict[str, Any],
    frame_id: int,
    n_frames: int,
) -> List[int]:
    raw = record.get("subspace_neighbour_frame_ids")
    if not isinstance(raw, list):
        raw = record.get("neighbour_frame_ids")
    neighbours: List[int] = []
    if isinstance(raw, list):
        for item in raw:
            nid = _safe_int(item)
            if nid is not None and 0 <= nid < int(n_frames) and nid != int(frame_id):
                neighbours.append(int(nid))
    if neighbours:
        return sorted(set(neighbours))
    return []


def _neighbour_count_from_config(config: Any) -> int:
    try:
        raw = config.acquisition.subspace.neighbour_count
        return max(1, min(16, int(raw)))
    except Exception:
        return 16


def _nearest_pool_neighbour_ids(
    pool: Any,
    seed_atoms: Atoms,
    frame_id: int,
    *,
    k: int,
) -> Tuple[List[int], Dict[str, Any]]:
    distances: List[Tuple[float, int]] = []
    n_frames = int(pool.n_frames())
    for candidate_id in range(n_frames):
        if int(candidate_id) == int(frame_id):
            continue
        try:
            distance = float(
                aligned_mass_weighted_rmsd(seed_atoms, pool.frame(int(candidate_id)))
            )
        except Exception:
            continue
        if _finite_positive(distance) is not None:
            distances.append((float(distance), int(candidate_id)))
    distances.sort(key=lambda item: (item[0], item[1]))
    picked = [int(candidate_id) for _, candidate_id in distances[: int(k)]]
    return picked, {
        "n_pool_frames": int(n_frames),
        "n_pool_frames_scanned": int(max(0, n_frames - 1)),
        "n_finite_positive_nearest_distances": int(len(distances)),
        "nearest_k": int(k),
    }


def _local_motion_values(
    campaign_dir: Path,
    iter_dir: Path,
    config: Any,
) -> Tuple[List[float], Dict[str, Any]]:
    from .acquisition.trajectory_pool import TrajectoryPool

    pool = TrajectoryPool.load(campaign_dir)
    records = _load_raw_seed_records(iter_dir)
    values: List[float] = []
    n_seed_records = 0
    n_neighbour_pairs = 0
    neighbour_source_counts: Dict[str, int] = {}
    nearest_diag = {
        "n_pool_frames": int(pool.n_frames()),
        "n_pool_frames_scanned": 0,
        "n_finite_positive_nearest_distances": 0,
        "nearest_k": _neighbour_count_from_config(config),
    }
    for rec in records:
        fid = _safe_int(rec.get("frame_id"))
        if fid is None:
            continue
        if not 0 <= int(fid) < pool.n_frames():
            continue
        n_seed_records += 1
        seed_atoms = pool.frame(int(fid))
        neighbour_ids = _manifest_neighbour_ids_for_seed(rec, int(fid), pool.n_frames())
        neighbour_source = "manifest_ids"
        if not neighbour_ids:
            neighbour_source = "nearest_pool_rmsd"
            neighbour_ids, diag = _nearest_pool_neighbour_ids(
                pool,
                seed_atoms,
                int(fid),
                k=nearest_diag["nearest_k"],
            )
            for key in (
                "n_pool_frames_scanned",
                "n_finite_positive_nearest_distances",
            ):
                nearest_diag[key] = int(nearest_diag.get(key, 0)) + int(diag.get(key, 0))
        neighbour_source_counts[neighbour_source] = (
            int(neighbour_source_counts.get(neighbour_source, 0)) + 1
        )
        for nid in neighbour_ids:
            try:
                d = float(aligned_mass_weighted_rmsd(seed_atoms, pool.frame(int(nid))))
            except Exception:
                continue
            if _finite_positive(d) is not None:
                values.append(float(d))
                n_neighbour_pairs += 1
    return values, {
        "n_seed_records": int(n_seed_records),
        "n_neighbour_pairs": int(n_neighbour_pairs),
        "neighbour_sources": neighbour_source_counts,
        "nearest_pool_rmsd": nearest_diag,
    }


def _movement_from_landing_safety(safety: Any) -> Optional[float]:
    if not isinstance(safety, dict) or not bool(safety.get("accepted", False)):
        return None
    metrics = safety.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    for key in ("movement_rmsd_ang", "aligned_mass_weighted_rmsd_ang"):
        value = _finite_positive(metrics.get(key))
        if value is not None:
            return float(value)
    for key in ("movement_rmsd_ang", "aligned_mass_weighted_rmsd_ang"):
        value = _finite_positive(safety.get(key))
        if value is not None:
            return float(value)
    return None


def _movement_values_from_records(records: Any) -> Tuple[List[float], Dict[str, int]]:
    values: List[float] = []
    skipped_rejected = 0
    skipped_nonfinite = 0
    if not isinstance(records, list):
        return values, {
            "n_records": 0,
            "n_accepted_movements": 0,
            "n_skipped_rejected": 0,
            "n_skipped_nonfinite": 0,
        }
    for rec in records:
        if not isinstance(rec, dict):
            continue
        safety = rec.get("landing_safety")
        if not isinstance(safety, dict):
            skipped_nonfinite += 1
            continue
        if not bool(safety.get("accepted", False)):
            skipped_rejected += 1
            continue
        value = _movement_from_landing_safety(safety)
        if value is None:
            skipped_nonfinite += 1
            continue
        values.append(float(value))
    return values, {
        "n_records": int(len(records)),
        "n_accepted_movements": int(len(values)),
        "n_skipped_rejected": int(skipped_rejected),
        "n_skipped_nonfinite": int(skipped_nonfinite),
    }


def _ariadne_movement_history_values(
    campaign_dir: Path,
    iteration: int,
    window: int,
) -> Tuple[List[float], Dict[str, Any]]:
    if int(window) <= 0 or int(iteration) <= 0:
        return [], {"n_history_files": 0}
    base = campaign_dir / "7_ACTIVE_LEARNING"
    start = max(0, int(iteration) - int(window))
    values: List[float] = []
    n_files = 0
    n_audit_files = 0
    n_results_files = 0
    n_skipped_rejected = 0
    n_skipped_nonfinite = 0
    for previous in range(start, int(iteration)):
        iter_dir = base / ("iteration-" + str(previous).zfill(4))
        audit_path = iter_dir / "ARIADNE_LANDING_AUDIT.json"
        results_path = iter_dir / "ARIADNE_RESULTS.json"
        source_records = None
        source_kind = None
        try:
            if audit_path.is_file():
                payload = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    source_records = payload.get("seeds")
                    source_kind = "audit"
                    n_audit_files += 1
            elif results_path.is_file():
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    source_records = payload.get("accepted")
                    source_kind = "results"
                    n_results_files += 1
        except Exception:
            continue
        if source_records is None:
            continue
        new_values, diag = _movement_values_from_records(source_records)
        if not new_values and source_kind == "audit" and results_path.is_file():
            try:
                payload = json.loads(results_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                fallback_values, fallback_diag = _movement_values_from_records(
                    payload.get("accepted")
                )
                if fallback_values:
                    new_values = fallback_values
                    diag = fallback_diag
                    source_kind = "results"
                    n_results_files += 1
        values.extend(new_values)
        n_skipped_rejected += int(diag.get("n_skipped_rejected", 0))
        n_skipped_nonfinite += int(diag.get("n_skipped_nonfinite", 0))
        n_files += 1
    if n_audit_files and n_results_files:
        source = "mixed"
    elif n_audit_files:
        source = "ariadne_landing_audit"
    elif n_results_files:
        source = "ariadne_results"
    else:
        source = "none"
    return values, {
        "source": source,
        "n_history_files": int(n_files),
        "n_audit_files": int(n_audit_files),
        "n_results_files": int(n_results_files),
        "n_accepted_movements": int(len(values)),
        "n_skipped_rejected": int(n_skipped_rejected),
        "n_skipped_nonfinite": int(n_skipped_nonfinite),
    }


def _sidecar_provenance(campaign_dir: Path, iter_dir: Path, iteration: int) -> Dict[str, Any]:
    seed_path = iter_dir / "seeds_picked.json"
    seed_payload = _load_seed_payload(iter_dir)
    state_path = campaign_dir / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    training_version = None
    models_version = None
    state_available = False
    if state_path.is_file():
        try:
            state = read_state(state_path)
            training_version = int(getattr(state, "training_set_version", -1))
            models_version = int(getattr(state, "models_version", -1))
            state_available = True
        except Exception:
            state_available = False
    return {
        "campaign_dir": str(campaign_dir.resolve()),
        "iteration": int(iteration),
        "trajectory_sha256": str(seed_payload.get("trajectory_sha256") or ""),
        "seed_selection_manifest": (
            str(seed_path.resolve()) if seed_path.is_file() else None
        ),
        "seed_selection_sha256": _sha256_file(seed_path),
        "training_set_version": training_version,
        "models_version": models_version,
        "state_available": bool(state_available),
    }


def _normalise_config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(getattr(config, "geometry_novelty", object()), name, default)


def compute_geometry_novelty_scale(
    campaign_dir: Union[str, Path],
    config: Any,
    *,
    iteration: int,
) -> Dict[str, Any]:
    """Compute the schema-v1 geometry novelty scale payload.

    The result is intentionally diagnostic-rich. A fallback scale is still a
    valid payload, but it is labelled so operators can tell whether Phase B was
    using campaign data or the conservative configured default.
    """
    campaign = Path(campaign_dir)
    iter_dir = campaign / "7_ACTIVE_LEARNING" / ("iteration-" + str(int(iteration)).zfill(4))
    enabled = bool(_normalise_config_value(config, "enabled", True))
    scale_source = str(_normalise_config_value(config, "scale_source", "local_motion"))
    statistic = str(_normalise_config_value(config, "statistic", "median"))
    floor_value = float(_normalise_config_value(config, "scale_floor_angstrom", 1.0e-3))
    fallback_value = float(_normalise_config_value(config, "fallback_scale_angstrom", 0.05))
    history_window = int(_normalise_config_value(config, "history_window_iterations", 5))
    score_transform = GEOMETRY_NOVELTY_SCORE_TRANSFORM

    local_values: List[float] = []
    history_values: List[float] = []
    diagnostics: Dict[str, Any] = {}
    reasons: List[str] = []

    if enabled and scale_source in ("local_motion", "hybrid"):
        try:
            local_values, local_diag = _local_motion_values(campaign, iter_dir, config)
            diagnostics["local_motion"] = local_diag
        except Exception as exc:
            diagnostics["local_motion"] = {
                "error": type(exc).__name__ + ": " + str(exc),
            }
            reasons.append("local_motion_unavailable")

    if enabled and scale_source in ("movement_history", "hybrid"):
        try:
            history_values, history_diag = _ariadne_movement_history_values(
                campaign, int(iteration), history_window,
            )
            diagnostics["movement_history"] = history_diag
        except Exception as exc:
            diagnostics["movement_history"] = {
                "error": type(exc).__name__ + ": " + str(exc),
            }
            reasons.append("movement_history_unavailable")

    combined_values = [
        float(v)
        for v in [*local_values, *history_values]
        if _finite_positive(v) is not None
    ]
    fallback_used = False
    if enabled and combined_values:
        raw_scale = _percentile(combined_values, statistic)
        if _finite_positive(raw_scale) is None:
            fallback_used = True
            reasons.append("computed_scale_non_positive")
            raw_scale = float(fallback_value)
    else:
        fallback_used = True
        if enabled:
            reasons.append("no_motion_values")
        else:
            reasons.append("geometry_novelty_disabled")
        raw_scale = float(fallback_value)

    scale = max(float(raw_scale), float(floor_value))
    if scale != float(raw_scale):
        reasons.append("scale_floor_applied")
    scale_resolution_mode = (
        "fallback_protocol" if bool(fallback_used) else "computed"
    )

    return {
        "schema_version": GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION,
        "iteration": int(iteration),
        "generated_at_iso": _now_iso(),
        "enabled": bool(enabled),
        "scale_source": scale_source,
        "statistic": statistic,
        "score_transform": score_transform,
        "scale_floor_angstrom": float(floor_value),
        "history_window_iterations": int(history_window),
        "fallback_scale_angstrom": float(fallback_value),
        "scale_angstrom": float(scale),
        "raw_scale_angstrom": float(raw_scale),
        "fallback_used": bool(fallback_used),
        "scale_resolution_mode": scale_resolution_mode,
        "reasons": [str(reason) for reason in reasons],
        "n_values": int(len(combined_values)),
        "value_summary_angstrom": _summary(combined_values),
        "local_motion_summary_angstrom": _summary(local_values),
        "movement_history_summary_angstrom": _summary(history_values),
        "diagnostics": diagnostics,
        "provenance": _sidecar_provenance(campaign, iter_dir, int(iteration)),
    }


def write_geometry_novelty_scale(iter_dir: Union[str, Path], payload: Dict[str, Any]) -> Path:
    path = geometry_novelty_scale_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(payload))
    return path


def read_geometry_novelty_scale(
    iter_dir: Union[str, Path],
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = geometry_novelty_scale_path(iter_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("geometry novelty scale manifest must be a JSON object")
    schema = int(payload.get("schema_version", -1))
    if schema != GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION:
        raise ValueError(
            "geometry novelty scale schema_version "
            + str(schema)
            + " != "
            + str(GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION)
        )
    scale = _finite_positive(payload.get("scale_angstrom"))
    if scale is None:
        raise ValueError("geometry novelty scale_angstrom must be finite and > 0")
    if expected_iteration is not None and "iteration" in payload:
        try:
            iteration = int(payload.get("iteration"))
        except (TypeError, ValueError) as exc:
            raise ValueError("geometry novelty iteration must be an integer") from exc
        if iteration != int(expected_iteration):
            raise ValueError(
                "geometry novelty iteration mismatch: expected "
                + str(int(expected_iteration))
                + " got "
                + str(iteration)
            )
    return payload


def _campaign_iteration_dir(campaign_dir: Union[str, Path], iteration: int) -> Path:
    return (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )


def _config_enabled(config: Any) -> bool:
    return bool(_normalise_config_value(config, "enabled", True))


def _scale_from_payload(scale_payload: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(scale_payload, dict):
        return None
    return _finite_positive(scale_payload.get("scale_angstrom"))


def resolve_geometry_novelty_consumers(
    config: Any,
    scale_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve every current geometry-novelty consumer from one scale.

    This is the single HPC-side place where dimensionless campaign settings
    become Angstrom values. If geometry novelty is disabled, it reports values
    derived from the configured fallback scale.
    """
    enabled = _config_enabled(config)
    scale = _scale_from_payload(scale_payload) if enabled else None
    mode = "scaled" if scale is not None else "absolute"
    scale_payload_dict = scale_payload if isinstance(scale_payload, dict) else {}
    fallback_protocol = (
        scale is None
        or not bool(enabled)
        or bool(scale_payload_dict.get("fallback_used", False))
    )
    scale_resolution_mode = (
        "fallback_protocol" if fallback_protocol else "computed"
    )
    phase_b_scaled = PHASE_B_MIN_SEPARATION_SCALE

    if scale is not None:
        movement_band = {
            "threshold_mode": "scaled",
            "scale_resolution_mode": scale_resolution_mode,
            "scale_angstrom": float(scale),
            "hard_min_angstrom": MOVEMENT_BAND_HARD_MIN_FRACTION * float(scale),
            "target_low_angstrom": MOVEMENT_BAND_TARGET_LOW_FRACTION * float(scale),
            "target_peak_angstrom": MOVEMENT_BAND_TARGET_PEAK_FRACTION * float(scale),
            "target_high_angstrom": MOVEMENT_BAND_TARGET_HIGH_FRACTION * float(scale),
            "hard_max_angstrom": MOVEMENT_BAND_HARD_MAX_FRACTION * float(scale),
            "fractions": movement_band_fractions(),
        }
        movement_utility = {
            "threshold_mode": "scaled",
            "scale_resolution_mode": scale_resolution_mode,
            "low_softness_angstrom": (
                MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION * float(scale)
            ),
            "high_softness_angstrom": (
                MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION * float(scale)
            ),
            "low_softness_fraction": MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION,
            "high_softness_fraction": MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION,
        }
        fullspace = {
            "threshold_mode": "scaled",
            "scale_resolution_mode": scale_resolution_mode,
            "rmsd_scale_angstrom": FULLSPACE_RMSD_SCALE_MULTIPLIER * float(scale),
            "rmsd_scale_multiplier": FULLSPACE_RMSD_SCALE_MULTIPLIER,
        }
        phase_b = {
            "threshold_mode": "scaled",
            "scale_resolution_mode": scale_resolution_mode,
            "scaled_threshold": phase_b_scaled,
            "min_separation_scaled": phase_b_scaled,
            "effective_min_separation_angstrom": phase_b_scaled * float(scale),
        }
    else:
        fallback = float(_normalise_config_value(config, "fallback_scale_angstrom", 0.05))
        movement_band = {
            "threshold_mode": "absolute",
            "scale_resolution_mode": scale_resolution_mode,
            "hard_min_angstrom": MOVEMENT_BAND_HARD_MIN_FRACTION * fallback,
            "target_low_angstrom": MOVEMENT_BAND_TARGET_LOW_FRACTION * fallback,
            "target_peak_angstrom": MOVEMENT_BAND_TARGET_PEAK_FRACTION * fallback,
            "target_high_angstrom": MOVEMENT_BAND_TARGET_HIGH_FRACTION * fallback,
            "hard_max_angstrom": MOVEMENT_BAND_HARD_MAX_FRACTION * fallback,
            "fractions": movement_band_fractions(),
        }
        movement_utility = {
            "threshold_mode": "absolute",
            "scale_resolution_mode": scale_resolution_mode,
            "low_softness_angstrom": MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION * fallback,
            "high_softness_angstrom": MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION * fallback,
            "low_softness_fraction": MOVEMENT_UTILITY_LOW_SOFTNESS_FRACTION,
            "high_softness_fraction": MOVEMENT_UTILITY_HIGH_SOFTNESS_FRACTION,
        }
        fullspace = {
            "threshold_mode": "absolute",
            "scale_resolution_mode": scale_resolution_mode,
            "rmsd_scale_angstrom": FULLSPACE_RMSD_SCALE_MULTIPLIER * fallback,
            "rmsd_scale_multiplier": FULLSPACE_RMSD_SCALE_MULTIPLIER,
        }
        phase_b = {
            "threshold_mode": "absolute",
            "scale_resolution_mode": scale_resolution_mode,
            "effective_min_separation_angstrom": phase_b_scaled * fallback,
            "scaled_threshold": phase_b_scaled,
            "min_separation_scaled": phase_b_scaled,
        }

    return {
        "schema_version": 1,
        "enabled": bool(enabled),
        "threshold_mode": mode,
        "scale_resolution_mode": scale_resolution_mode,
        "scale_angstrom": None if scale is None else float(scale),
        "phase_b": phase_b,
        "movement_band": movement_band,
        "movement_utility": movement_utility,
        "fullspace_confinement": fullspace,
    }


def ensure_geometry_novelty_scale(
    campaign_dir: Union[str, Path],
    config: Any,
    *,
    iteration: int,
) -> Dict[str, Any]:
    """Read or write the per-iteration geometry-novelty scale sidecar.

    The helper is idempotent for restarts, but it also backfills the optional
    resolved-consumer diagnostics when reading an older schema-v1 sidecar.
    """
    iter_dir = _campaign_iteration_dir(campaign_dir, iteration)
    path = geometry_novelty_scale_path(iter_dir)
    if path.is_file():
        try:
            payload = read_geometry_novelty_scale(
                iter_dir,
                expected_iteration=int(iteration),
            )
        except Exception:
            payload = compute_geometry_novelty_scale(
                campaign_dir,
                config,
                iteration=int(iteration),
            )
    else:
        payload = compute_geometry_novelty_scale(
            campaign_dir,
            config,
            iteration=int(iteration),
        )
    payload = dict(payload)
    payload["resolved_consumers"] = resolve_geometry_novelty_consumers(
        config,
        payload,
    )
    write_geometry_novelty_scale(iter_dir, payload)
    return payload


def effective_phase_b_min_separation(config: Any, scale_payload: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    resolved = resolve_geometry_novelty_consumers(config, scale_payload)
    phase_b = dict(resolved.get("phase_b") or {})
    return (
        float(phase_b.get("effective_min_separation_angstrom", 0.0)),
        str(phase_b.get("threshold_mode", resolved.get("threshold_mode", "absolute"))),
    )


def apply_geometry_novelty_to_acquisition_config(
    acquisition_config: Any,
    config: Any,
    scale_payload: Optional[Dict[str, Any]],
) -> Any:
    """Return an AcquisitionConfig with ARIADNE geometry scales resolved.

    ``ichor_core`` receives plain numeric values only; it does not know about
    campaign directories or sidecar files.
    """
    resolved = resolve_geometry_novelty_consumers(config, scale_payload)
    if str(resolved.get("threshold_mode")) != "scaled":
        return acquisition_config

    movement_band = dict(resolved["movement_band"])
    movement_utility = dict(resolved["movement_utility"])
    fullspace = dict(resolved["fullspace_confinement"])
    scale = float(resolved["scale_angstrom"])

    return replace(
        acquisition_config,
        movement_band=replace(
            acquisition_config.movement_band,
            hard_min_floor_ang=float(movement_band["hard_min_angstrom"]),
            target_low_floor_ang=float(movement_band["target_low_angstrom"]),
            target_peak_floor_ang=float(movement_band["target_peak_angstrom"]),
            target_high_cap_ang=float(movement_band["target_high_angstrom"]),
            hard_max_cap_ang=float(movement_band["hard_max_angstrom"]),
            geometry_novelty_scale_angstrom=scale,
        ),
        movement_utility=replace(
            acquisition_config.movement_utility,
            low_softness_ang=float(movement_utility["low_softness_angstrom"]),
            high_softness_ang=float(movement_utility["high_softness_angstrom"]),
        ),
        fullspace_confinement=replace(
            acquisition_config.fullspace_confinement,
            rmsd_scale_ang=float(fullspace["rmsd_scale_angstrom"]),
        ),
    )


def scaled_distances(distances_angstrom: Iterable[Any], scale_angstrom: Any) -> List[Optional[float]]:
    scale = _finite_positive(scale_angstrom)
    if scale is None:
        return [None for _ in distances_angstrom]
    out: List[Optional[float]] = []
    for value in distances_angstrom:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if not math.isfinite(distance):
            out.append(None)
        else:
            out.append(float(distance / scale))
    return out


def novelty_score(distance_angstrom: Any, scale_angstrom: Any, transform: str = "linear_cap") -> Optional[float]:
    scale = _finite_positive(scale_angstrom)
    if scale is None:
        return None
    try:
        distance = float(distance_angstrom)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(distance):
        return 1.0 if distance > 0.0 else None
    if distance < 0.0:
        return None
    ratio = float(distance / scale)
    if transform == "exponential":
        return float(1.0 - math.exp(-max(0.0, ratio)))
    return float(min(1.0, max(0.0, ratio)))
