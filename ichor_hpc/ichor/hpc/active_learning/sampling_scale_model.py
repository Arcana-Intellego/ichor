"""Daemon-owned scale model for the public sampling protocol.

The campaign exposes one sampling aggressiveness value.  This module derives
the dimensional scales used to turn that public value into concrete ARIADNE
and Phase-B thresholds.  It deliberately uses only data the daemon already
produces: geometry-novelty sidecars, previous ARIADNE landing audits, and
per-seed result JSON files.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .daemon.state import atomic_write_json
from .geometry_protocol import FULLSPACE_RMSD_SCALE_MULTIPLIER


SAMPLING_SCALE_MODEL_SCHEMA_VERSION = 1
SAMPLING_SCALE_MODEL_FILENAME = "SAMPLING_SCALE_MODEL.json"


def sampling_scale_model_path(iter_dir: Union[str, Path]) -> Path:
    return Path(iter_dir) / SAMPLING_SCALE_MODEL_FILENAME


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


def _finite_nonnegative(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0.0:
        return None
    return out


def _percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    clean = sorted(float(v) for v in values if _finite_positive(v) is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    pos = max(0.0, min(1.0, float(fraction))) * float(len(clean) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(clean[lo])
    w = pos - float(lo)
    return float((1.0 - w) * clean[lo] + w * clean[hi])


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
        "p25": _percentile(clean, 0.25),
        "median": _percentile(clean, 0.50),
        "p75": _percentile(clean, 0.75),
        "max": float(max(clean)),
    }


def _json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _records_from_previous_iterations(
    campaign_dir: Path,
    iteration: int,
    *,
    window: int,
) -> Iterable[Tuple[Path, Dict[str, Any]]]:
    if int(iteration) <= 0 or int(window) <= 0:
        return
    base = campaign_dir / "7_ACTIVE_LEARNING"
    start = max(0, int(iteration) - int(window))
    for previous in range(start, int(iteration)):
        iter_dir = base / ("iteration-" + str(previous).zfill(4))
        audit = _json(iter_dir / "ARIADNE_LANDING_AUDIT.json")
        if isinstance(audit, dict):
            for record in audit.get("seeds") or []:
                if isinstance(record, dict):
                    yield iter_dir, record
            continue
        results = _json(iter_dir / "ARIADNE_RESULTS.json")
        if isinstance(results, dict):
            for record in results.get("accepted") or []:
                if isinstance(record, dict):
                    yield iter_dir, record


def _record_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    safety = record.get("landing_safety")
    if isinstance(safety, dict) and isinstance(safety.get("metrics"), dict):
        return dict(safety.get("metrics") or {})
    if isinstance(record.get("metrics"), dict):
        return dict(record.get("metrics") or {})
    return {}


def _result_path(iter_dir: Path, record: Dict[str, Any]) -> Optional[Path]:
    raw = record.get("result_json")
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = iter_dir / path
        return path
    seed_dir = record.get("seed_dir")
    if seed_dir:
        path = Path(str(seed_dir))
        if not path.is_absolute():
            path = iter_dir / path
        return path / "result.json"
    return None


def _coords(payload: Dict[str, Any], *names: str) -> Optional[List[List[float]]]:
    for name in names:
        raw = payload.get(name)
        if not isinstance(raw, list):
            continue
        rows: List[List[float]] = []
        ok = True
        for row in raw:
            if not isinstance(row, list) or len(row) != 3:
                ok = False
                break
            vals: List[float] = []
            for value in row:
                try:
                    out = float(value)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not math.isfinite(out):
                    ok = False
                    break
                vals.append(out)
            if not ok:
                break
            rows.append(vals)
        if ok and rows:
            return rows
    return None


def _per_atom_displacements(result: Dict[str, Any]) -> List[float]:
    start = _coords(result, "seed_coordinates", "initial_coordinates")
    final = _coords(result, "final_coordinates")
    if start is None or final is None or len(start) != len(final):
        return []
    out: List[float] = []
    for a, b in zip(start, final):
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        dz = float(b[2]) - float(a[2])
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if math.isfinite(d):
            out.append(float(d))
    return out


def _collect_history(
    campaign_dir: Path,
    iteration: int,
    *,
    window: int,
) -> Dict[str, Any]:
    movement_values: List[float] = []
    rmsd_values: List[float] = []
    residual_values: List[float] = []
    max_atom_displacements: List[float] = []
    min_pair_distances: List[float] = []
    per_atom_by_index: Dict[int, List[float]] = {}
    n_records = 0
    n_result_json = 0
    for iter_dir, record in _records_from_previous_iterations(
        campaign_dir,
        iteration,
        window=window,
    ):
        n_records += 1
        metrics = _record_metrics(record)
        for key, dest in (
            ("movement_rmsd_ang", movement_values),
            ("aligned_mass_weighted_rmsd_ang", movement_values),
            ("aligned_rmsd_ang", rmsd_values),
            ("fullspace_residual_distance", residual_values),
            ("max_displacement_ang", max_atom_displacements),
            ("min_pair_distance_ang", min_pair_distances),
        ):
            value = _finite_positive(metrics.get(key))
            if value is not None:
                dest.append(float(value))
        result_path = _result_path(iter_dir, record)
        if result_path is None or not result_path.is_file():
            continue
        result = _json(result_path)
        if not isinstance(result, dict):
            continue
        n_result_json += 1
        for index, displacement in enumerate(_per_atom_displacements(result)):
            if _finite_nonnegative(displacement) is not None:
                per_atom_by_index.setdefault(int(index), []).append(float(displacement))
    return {
        "n_records": int(n_records),
        "n_result_json": int(n_result_json),
        "movement_values": movement_values,
        "rmsd_values": rmsd_values,
        "residual_values": residual_values,
        "max_atom_displacements": max_atom_displacements,
        "min_pair_distances": min_pair_distances,
        "per_atom_by_index": per_atom_by_index,
    }


def _scale_entry(
    value: Optional[float],
    *,
    source: str,
    fallback: float,
    n: int,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    finite = _finite_positive(value)
    used_fallback = finite is None
    out = float(fallback if finite is None else finite)
    return {
        "value_angstrom": out,
        "source": "fallback" if used_fallback else str(source),
        "n_values": int(n),
        "fallback_used": bool(used_fallback),
        "fallback_reason": fallback_reason if used_fallback else None,
    }


def build_sampling_scale_model(
    campaign_dir: Union[str, Path],
    config: Any,
    iteration: int,
    *,
    geometry_scale_payload: Optional[Dict[str, Any]] = None,
    write_manifest: bool = True,
) -> Dict[str, Any]:
    """Build the daemon-owned scale model for one active-learning iteration."""
    campaign = Path(campaign_dir)
    iter_dir = campaign / "7_ACTIVE_LEARNING" / ("iteration-" + str(int(iteration)).zfill(4))
    fallback_geometry = _finite_positive(
        getattr(getattr(config, "geometry_novelty", None), "fallback_scale_angstrom", None)
    ) or 0.05
    geometry_payload = dict(geometry_scale_payload or {})
    geometry_value = _finite_positive(geometry_payload.get("scale_angstrom"))
    geometry_source = str(geometry_payload.get("scale_resolution_mode") or "geometry_novelty")
    geometry_n = int(geometry_payload.get("n_values") or 0)
    window = int(
        getattr(getattr(config, "geometry_novelty", None), "history_window_iterations", 5)
        or 5
    )
    history = _collect_history(campaign, int(iteration), window=window)
    movement_median = _percentile(history["movement_values"], 0.50)
    if movement_median is not None:
        geometry_value = movement_median
        geometry_source = "ariadne_landing_history"
        geometry_n = len(history["movement_values"])

    geometry_scale = _scale_entry(
        geometry_value,
        source=geometry_source,
        fallback=fallback_geometry,
        n=geometry_n,
        fallback_reason="no_finite_geometry_motion_values",
    )
    geometry_ang = float(geometry_scale["value_angstrom"])
    rmsd_fallback = max(geometry_ang, 1.0e-3)
    residual_fallback = max(
        geometry_ang * float(FULLSPACE_RMSD_SCALE_MULTIPLIER),
        1.0e-3,
    )
    aligned_rmsd = _scale_entry(
        _percentile(history["rmsd_values"] or history["movement_values"], 0.50),
        source="ariadne_landing_history",
        fallback=rmsd_fallback,
        n=len(history["rmsd_values"] or history["movement_values"]),
        fallback_reason="no_finite_aligned_rmsd_history",
    )
    residual = _scale_entry(
        _percentile(history["residual_values"], 0.50),
        source="ariadne_landing_history",
        fallback=residual_fallback,
        n=len(history["residual_values"]),
        fallback_reason="no_finite_residual_history",
    )

    per_atom_values: List[float] = []
    per_atom_counts: List[int] = []
    for index in sorted(history["per_atom_by_index"]):
        values = history["per_atom_by_index"][index]
        value = _percentile(values, 0.50)
        if value is not None:
            per_atom_values.append(max(float(value), 1.0e-3))
            per_atom_counts.append(int(len(values)))
    if not per_atom_values:
        per_atom_values = [geometry_ang]
        per_atom_counts = [0]
        per_atom_source = "uniform_geometry_motion_scale"
        per_atom_fallback = True
    else:
        per_atom_source = "result_json_per_atom_displacement_history"
        per_atom_fallback = False

    min_pair_floor = _finite_positive(
        getattr(getattr(config, "quality_gates", None), "ariadne_min_pair_distance_ang", None)
    ) or 0.60
    observed_pair = _percentile(history["min_pair_distances"], 0.50)
    pair_reference = min_pair_floor if observed_pair is None else max(min_pair_floor, float(observed_pair))
    ratio_floor = float(min_pair_floor) / float(pair_reference)

    gradient_scale = {
        "rms_acquisition_gradient": {
            "value": _finite_positive(
                getattr(getattr(config, "ariadne", None), "trqn_target_initial_grad_rms", None)
            ) or 2.0e-4,
            "source": "resolved_ariadne_target_initial_grad_rms",
            "fallback_used": False,
        }
    }

    payload = {
        "schema_version": SAMPLING_SCALE_MODEL_SCHEMA_VERSION,
        "iteration": int(iteration),
        "generated_at_iso": _now_iso(),
        "geometry_motion_scale": geometry_scale,
        "aligned_rmsd_scale": aligned_rmsd,
        "residual_fullspace_scale": residual,
        "per_atom_mobility_scales": {
            "mode": "per_atom_index" if not per_atom_fallback else "uniform",
            "values_angstrom": per_atom_values,
            "counts": per_atom_counts,
            "source": per_atom_source,
            "fallback_used": bool(per_atom_fallback),
            "floor_angstrom": 1.0e-3,
        },
        "pair_distance_reference": {
            "mode": "minimum_safe_pair_distance_ratio",
            "reference_min_pair_distance_angstrom": float(pair_reference),
            "hard_floor_angstrom": float(min_pair_floor),
            "ratio_floor": float(ratio_floor),
            "source": (
                "ariadne_landing_history"
                if observed_pair is not None
                else "quality_gate_hard_floor"
            ),
            "fallback_used": observed_pair is None,
            "observed_min_pair_summary": _summary(history["min_pair_distances"]),
        },
        "bond_angle_reference": {
            "source": "not_available_wave2",
            "fallback_used": True,
        },
        "gradient_rms_scale": gradient_scale,
        "component_scales": {
            "source": "existing_reference_scales_only",
            "fallback_used": True,
        },
        "history": {
            "window_iterations": int(window),
            "n_records": int(history["n_records"]),
            "n_result_json": int(history["n_result_json"]),
            "movement_summary": _summary(history["movement_values"]),
            "aligned_rmsd_summary": _summary(history["rmsd_values"]),
            "residual_summary": _summary(history["residual_values"]),
            "max_atom_displacement_summary": _summary(history["max_atom_displacements"]),
        },
        "diagnostics": {
            "size_independence_wave": 2,
            "new_scheduler_jobs": 0,
            "uses_only_existing_campaign_data": True,
        },
    }
    if write_manifest:
        iter_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(sampling_scale_model_path(iter_dir), payload)
    return payload


def read_sampling_scale_model(
    iter_dir: Union[str, Path],
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = sampling_scale_model_path(iter_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SAMPLING_SCALE_MODEL.json must contain an object")
    if int(payload.get("schema_version", -1)) != SAMPLING_SCALE_MODEL_SCHEMA_VERSION:
        raise ValueError("unsupported sampling scale model schema")
    if expected_iteration is not None and int(payload.get("iteration", -1)) != int(expected_iteration):
        raise ValueError("sampling scale model iteration mismatch")
    return payload


def scale_model_value(payload: Optional[Dict[str, Any]], path: Sequence[str], default: Optional[float] = None) -> Optional[float]:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(str(key))
    return _finite_positive(cur) if cur is not None else default


__all__ = [
    "SAMPLING_SCALE_MODEL_FILENAME",
    "SAMPLING_SCALE_MODEL_SCHEMA_VERSION",
    "build_sampling_scale_model",
    "read_sampling_scale_model",
    "sampling_scale_model_path",
    "scale_model_value",
]
