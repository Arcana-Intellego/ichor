"""AIMAll science-quality sidecar generation.

The acceptance manifest is intentionally kept small and stable. This module
records the richer AIMAll/IQA checks in a sibling JSON manifest that can evolve
without changing downstream handoff schemas.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .state import atomic_write_json


QUANTUM_QUALITY_MANIFEST = "quantum_quality.json"
QUANTUM_QUALITY_SCHEMA_VERSION = 1


@dataclass
class _DefaultQualityGates:
    require_readable_aimall_geometry: bool = True
    require_finite_iqa: bool = True
    require_finite_integration_error: bool = True
    max_abs_integration_error: Optional[float] = None
    iqa_energy_recovery_tolerance_ha: Optional[float] = None


def _gate_value(gates: Any, name: str) -> Any:
    return getattr(gates, name, getattr(_DefaultQualityGates(), name))


def _finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def evaluate_aimall_pointdir(pointdir: Any, gates: Any = None) -> Dict[str, Any]:
    """Return quality metrics and rejection reasons for one AIMAll pointdir."""
    gates = gates or _DefaultQualityGates()
    path = Path(getattr(pointdir, "path", pointdir))
    reasons: List[str] = []

    atom_count = None
    try:
        atom_count = int(len(pointdir.atoms))
    except Exception:
        if bool(_gate_value(gates, "require_readable_aimall_geometry")):
            reasons.append("aimall_geometry_unreadable")

    int_files = []
    try:
        ints = getattr(pointdir, "ints", None)
        int_files = list(getattr(ints, "ints", [])) if ints is not None else []
    except Exception:
        reasons.append("int_directory_parse_failure")
        int_files = []
    if not int_files:
        reasons.append("no_int_files")
    if atom_count is not None and int_files and len(int_files) != atom_count:
        reasons.append(
            "aimall_partial_" + str(len(int_files)) + "_of_" + str(atom_count) + "_int"
        )

    per_atom: List[Dict[str, Any]] = []
    iqa_values: List[float] = []
    integration_values: List[float] = []
    for int_file in int_files:
        atom_name = None
        iqa = None
        integration_error = None
        atom_reasons: List[str] = []
        try:
            atom_name = str(getattr(int_file, "atom_name"))
        except Exception:
            atom_reasons.append("atom_name_unreadable")
        try:
            iqa = _finite_float(getattr(int_file, "iqa"))
        except Exception:
            iqa = None
        if iqa is None:
            atom_reasons.append("iqa_missing_or_nonfinite")
            if bool(_gate_value(gates, "require_finite_iqa")):
                reasons.append("iqa_missing_or_nonfinite")
        else:
            iqa_values.append(iqa)
        try:
            integration_error = _finite_float(getattr(int_file, "integration_error"))
        except Exception:
            integration_error = None
        if integration_error is None:
            atom_reasons.append("integration_error_missing_or_nonfinite")
            if bool(_gate_value(gates, "require_finite_integration_error")):
                reasons.append("integration_error_missing_or_nonfinite")
        else:
            integration_values.append(integration_error)
        per_atom.append(
            {
                "atom": atom_name,
                "iqa_ha": iqa,
                "integration_error": integration_error,
                "reasons": atom_reasons,
            }
        )

    max_abs_integration = (
        max(abs(v) for v in integration_values) if integration_values else None
    )
    configured_max_abs = _gate_value(gates, "max_abs_integration_error")
    configured_max_abs = _finite_float(configured_max_abs)
    if (
        configured_max_abs is not None
        and max_abs_integration is not None
        and max_abs_integration > configured_max_abs
    ):
        reasons.append("integration_error_threshold_exceeded")

    wfn_total_energy = None
    try:
        wfn = getattr(pointdir, "wfn", None)
        if wfn is not None:
            wfn_total_energy = _finite_float(getattr(wfn, "total_energy"))
    except Exception:
        wfn_total_energy = None

    sum_iqa = sum(iqa_values) if iqa_values else None
    recovery_error = (
        float(sum_iqa - wfn_total_energy)
        if sum_iqa is not None and wfn_total_energy is not None
        else None
    )
    recovery_tol = _finite_float(_gate_value(gates, "iqa_energy_recovery_tolerance_ha"))
    if (
        recovery_tol is not None
        and recovery_error is not None
        and abs(recovery_error) > recovery_tol
    ):
        reasons.append("iqa_energy_recovery_threshold_exceeded")

    deduped_reasons = sorted(set(reasons))
    return {
        "pointdir": path.name,
        "path": str(path.resolve()),
        "accepted": not deduped_reasons,
        "reasons": deduped_reasons,
        "atom_count": atom_count,
        "n_int": int(len(int_files)),
        "sum_iqa_ha": sum_iqa,
        "wfn_total_energy_ha": wfn_total_energy,
        "iqa_energy_recovery_error_ha": recovery_error,
        "max_abs_integration_error": max_abs_integration,
        "per_atom": per_atom,
    }


def write_quantum_quality_manifest(
    staging_dir: Path,
    *,
    phase_name: str,
    iteration: int,
    records: Sequence[Dict[str, Any]],
    gates: Any,
) -> Path:
    staging = Path(staging_dir)
    staging.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": QUANTUM_QUALITY_SCHEMA_VERSION,
        "phase": str(phase_name),
        "iteration": int(iteration),
        "thresholds": asdict(gates) if hasattr(gates, "__dataclass_fields__") else {},
        "n_total": int(len(records)),
        "n_accepted": int(sum(1 for r in records if bool(r.get("accepted")))),
        "n_rejected": int(sum(1 for r in records if not bool(r.get("accepted")))),
        "records": list(records),
    }
    path = staging / QUANTUM_QUALITY_MANIFEST
    atomic_write_json(path, payload)
    return path
