"""Geometry novelty scaling for Phase-B anti-overlap.

The Phase-B de-duplication metric is an aligned RMSD-like distance in
Angstrom. A single absolute threshold is awkward across differently sized or
flexible systems, so this module estimates a per-iteration motion scale and
lets Phase B express its threshold as a dimensionless multiple of that scale.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ichor.core.adversarial.geometry import aligned_mass_weighted_rmsd
from ichor.core.atoms import Atoms

from .daemon.state import atomic_write_json


GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION = 1
GEOMETRY_NOVELTY_SCALE_FILENAME = "GEOMETRY_NOVELTY_SCALE.json"
EXACT_DUPLICATE_EPSILON_ANGSTROM = 1.0e-12


__all__ = [
    "EXACT_DUPLICATE_EPSILON_ANGSTROM",
    "GEOMETRY_NOVELTY_SCALE_FILENAME",
    "GEOMETRY_NOVELTY_SCALE_SCHEMA_VERSION",
    "compute_geometry_novelty_scale",
    "effective_phase_b_min_separation",
    "geometry_novelty_scale_path",
    "novelty_score",
    "read_geometry_novelty_scale",
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


def _load_raw_seed_records(iter_dir: Path) -> List[Dict[str, Any]]:
    path = iter_dir / "seeds_picked.json"
    data = json.loads(path.read_text(encoding="utf-8"))
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


def _neighbour_ids_for_seed(record: Dict[str, Any], frame_id: int, n_frames: int) -> List[int]:
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

    adjacent: List[int] = []
    if int(frame_id) - 1 >= 0:
        adjacent.append(int(frame_id) - 1)
    if int(frame_id) + 1 < int(n_frames):
        adjacent.append(int(frame_id) + 1)
    return adjacent


def _local_motion_values(campaign_dir: Path, iter_dir: Path) -> Tuple[List[float], Dict[str, Any]]:
    from .acquisition.trajectory_pool import TrajectoryPool

    pool = TrajectoryPool.load(campaign_dir)
    records = _load_raw_seed_records(iter_dir)
    values: List[float] = []
    n_seed_records = 0
    n_neighbour_pairs = 0
    for rec in records:
        fid = _safe_int(rec.get("frame_id"))
        if fid is None:
            continue
        if not 0 <= int(fid) < pool.n_frames():
            continue
        n_seed_records += 1
        seed_atoms = pool.frame(int(fid))
        for nid in _neighbour_ids_for_seed(rec, int(fid), pool.n_frames()):
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
    }


def _history_values(campaign_dir: Path, iteration: int, window: int) -> Tuple[List[float], Dict[str, Any]]:
    if int(window) <= 0 or int(iteration) <= 0:
        return [], {"n_history_files": 0}
    base = campaign_dir / "7_ACTIVE_LEARNING"
    start = max(0, int(iteration) - int(window))
    values: List[float] = []
    n_files = 0
    for previous in range(start, int(iteration)):
        path = geometry_novelty_scale_path(
            base / ("iteration-" + str(previous).zfill(4))
        )
        if not path.is_file():
            continue
        try:
            payload = read_geometry_novelty_scale(path.parent)
        except Exception:
            continue
        val = _finite_positive(payload.get("scale_angstrom"))
        if val is None:
            continue
        values.append(float(val))
        n_files += 1
    return values, {"n_history_files": int(n_files)}


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
    score_transform = str(_normalise_config_value(config, "score_transform", "linear_cap"))

    local_values: List[float] = []
    history_values: List[float] = []
    diagnostics: Dict[str, Any] = {}
    reasons: List[str] = []

    if enabled and scale_source in ("local_motion", "hybrid"):
        try:
            local_values, local_diag = _local_motion_values(campaign, iter_dir)
            diagnostics["local_motion"] = local_diag
        except Exception as exc:
            diagnostics["local_motion"] = {
                "error": type(exc).__name__ + ": " + str(exc),
            }
            reasons.append("local_motion_unavailable")

    if enabled and scale_source in ("movement_history", "hybrid"):
        try:
            history_values, history_diag = _history_values(
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
        "reasons": [str(reason) for reason in reasons],
        "n_values": int(len(combined_values)),
        "value_summary_angstrom": _summary(combined_values),
        "local_motion_summary_angstrom": _summary(local_values),
        "movement_history_summary_angstrom": _summary(history_values),
        "diagnostics": diagnostics,
    }


def write_geometry_novelty_scale(iter_dir: Union[str, Path], payload: Dict[str, Any]) -> Path:
    path = geometry_novelty_scale_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, dict(payload))
    return path


def read_geometry_novelty_scale(iter_dir: Union[str, Path]) -> Dict[str, Any]:
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
    return payload


def effective_phase_b_min_separation(config: Any, scale_payload: Optional[Dict[str, Any]]) -> Tuple[float, str]:
    if bool(_normalise_config_value(config, "enabled", True)):
        if not isinstance(scale_payload, dict):
            raise ValueError("scaled Phase B threshold requested without a scale payload")
        scale = _finite_positive(scale_payload.get("scale_angstrom"))
        if scale is None:
            raise ValueError("scale_angstrom must be finite and > 0")
        scaled = float(getattr(config.phase_b, "min_separation_scaled", 0.5))
        return float(scaled * scale), "scaled"
    return float(getattr(config.phase_b, "min_separation", 0.0)), "absolute"


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
