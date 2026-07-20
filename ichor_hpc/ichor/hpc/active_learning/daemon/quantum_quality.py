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
from ..strict_json import load_path


QUANTUM_QUALITY_MANIFEST = "quantum_quality.json"
QUANTUM_QUALITY_SCHEMA_VERSION = 2
SUPPORTED_AIMALL_METHODS = frozenset({"HF", "M062X", "B3LYP", "PBE", "PBE0"})


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


def canonicalise_aimall_method(value: Any) -> str:
    """Return the electronic method token shared by WFN and INT contracts."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("AIMAll electronic method must be a non-empty string")
    compact = "".join(character for character in value.upper() if character.isalnum())
    for prefix in ("UNRESTRICTED", "RESTRICTED"):
        if compact.startswith(prefix):
            compact = compact[len(prefix) :]
            break
    aliases = {
        "HARTREEFOCK": "HF",
        "RHF": "HF",
        "UHF": "HF",
        "ROHF": "HF",
        "M062X": "M062X",
    }
    compact = aliases.get(compact, compact)
    if compact not in SUPPORTED_AIMALL_METHODS and len(compact) > 1:
        unrestricted = compact[1:] if compact[0] in {"R", "U"} else compact
        compact = aliases.get(unrestricted, unrestricted)
    if compact not in SUPPORTED_AIMALL_METHODS:
        raise ValueError("unsupported AIMAll electronic method: " + repr(value))
    return compact


def evaluate_aimall_pointdir(
    pointdir: Any,
    gates: Any = None,
    *,
    expected_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Return quality metrics and rejection reasons for one AIMAll pointdir."""
    gates = gates or _DefaultQualityGates()
    path = Path(getattr(pointdir, "path", pointdir))
    reasons: List[str] = []

    atom_count = None
    expected_atom_names: List[str] = []
    try:
        geometry_atoms = list(pointdir.atoms)
        atom_count = int(len(geometry_atoms))
        expected_atom_names = [str(atom.name) for atom in geometry_atoms]
        if not expected_atom_names or len(expected_atom_names) != len(
            set(expected_atom_names)
        ):
            raise ValueError("geometry atom identities are empty or duplicated")
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

    expected_method_canonical = (
        canonicalise_aimall_method(expected_method)
        if expected_method is not None
        else None
    )
    per_atom: List[Dict[str, Any]] = []
    iqa_values: List[float] = []
    integration_values: List[float] = []
    observed_methods: List[str] = []
    observed_atom_names: List[str] = []
    from ichor.core.common.constants import multipole_names

    required_multipoles = set(multipole_names)
    for int_file in int_files:
        atom_name = None
        iqa = None
        integration_error = None
        dft_model = None
        canonical_dft_model = None
        atom_reasons: List[str] = []
        try:
            atom_name = str(getattr(int_file, "atom_name"))
            observed_atom_names.append(atom_name)
            if Path(getattr(int_file, "path")).stem.capitalize() != atom_name:
                atom_reasons.append("int_filename_atom_mismatch")
                reasons.append("int_filename_atom_mismatch")
        except Exception:
            atom_reasons.append("atom_name_unreadable")
            reasons.append("atom_name_unreadable")
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
        try:
            dft_model = str(getattr(int_file, "dft_model"))
            canonical_dft_model = canonicalise_aimall_method(dft_model)
            observed_methods.append(canonical_dft_model)
        except Exception:
            atom_reasons.append("dft_model_missing_or_unsupported")
            reasons.append("dft_model_missing_or_unsupported")
        if (
            expected_method_canonical is not None
            and canonical_dft_model is not None
            and canonical_dft_model != expected_method_canonical
        ):
            atom_reasons.append("dft_model_mismatch")
            reasons.append("dft_model_mismatch")
        multipoles: Dict[str, Optional[float]] = {}
        try:
            raw_multipoles = dict(getattr(int_file, "global_spherical_multipoles"))
            if not required_multipoles.issubset(set(raw_multipoles)):
                atom_reasons.append("multipole_key_set_mismatch")
                reasons.append("multipole_key_set_mismatch")
            for label in sorted(required_multipoles):
                value = _finite_float(raw_multipoles.get(label))
                multipoles[label] = value
                if value is None:
                    atom_reasons.append("multipole_missing_or_nonfinite:" + label)
                    reasons.append("multipole_missing_or_nonfinite")
        except Exception:
            atom_reasons.append("multipole_parse_failure")
            reasons.append("multipole_parse_failure")
        per_atom.append(
            {
                "atom": atom_name,
                "int_file": Path(getattr(int_file, "path", "")).name,
                "dft_model": dft_model,
                "canonical_dft_model": canonical_dft_model,
                "iqa_ha": iqa,
                "integration_error": integration_error,
                "multipoles": multipoles,
                "reasons": atom_reasons,
            }
        )

    if expected_atom_names:
        if len(observed_atom_names) != len(set(observed_atom_names)):
            reasons.append("duplicate_int_atom_identity")
        expected_set = set(expected_atom_names)
        observed_set = set(observed_atom_names)
        if observed_set != expected_set:
            reasons.append("int_atom_identity_mismatch")
        order = {name: index for index, name in enumerate(expected_atom_names)}
        per_atom.sort(key=lambda record: order.get(str(record.get("atom")), len(order)))

    unique_methods = sorted(set(observed_methods))
    if len(unique_methods) > 1:
        reasons.append("mixed_dft_models")

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
    wfn_virial_ratio = None
    try:
        wfn = getattr(pointdir, "wfn", None)
        if wfn is not None:
            wfn_total_energy = _finite_float(getattr(wfn, "total_energy"))
            wfn_virial_ratio = _finite_float(getattr(wfn, "virial_ratio"))
    except Exception:
        wfn_total_energy = None
        wfn_virial_ratio = None
    if wfn_total_energy is None:
        reasons.append("wfn_total_energy_missing_or_nonfinite")
    if wfn_virial_ratio is None:
        reasons.append("wfn_virial_ratio_missing_or_nonfinite")

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
        "accepted": not deduped_reasons,
        "reasons": deduped_reasons,
        "atom_count": atom_count,
        "expected_atom_names": expected_atom_names,
        "n_int": int(len(int_files)),
        "sum_iqa_ha": sum_iqa,
        "wfn_total_energy_ha": wfn_total_energy,
        "wfn_virial_ratio": wfn_virial_ratio,
        "iqa_energy_recovery_error_ha": recovery_error,
        "max_abs_integration_error": max_abs_integration,
        "expected_dft_model": expected_method_canonical,
        "observed_dft_models": unique_methods,
        "per_atom": per_atom,
    }


def write_quantum_quality_manifest(
    staging_dir: Path,
    *,
    phase_name: str,
    iteration: int,
    records: Sequence[Dict[str, Any]],
    gates: Any,
    manifest_path: Optional[Path] = None,
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
    path = (
        Path(manifest_path)
        if manifest_path is not None
        else staging / QUANTUM_QUALITY_MANIFEST
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def read_quantum_quality_manifest(
    staging_dir: Path,
    *,
    expected_phase: str,
    expected_iteration: int,
    expected_pointdirs: Optional[Sequence[str]] = None,
    expected_method: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Read quality evidence and bind every record to one staged pointdir."""
    path = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(staging_dir) / QUANTUM_QUALITY_MANIFEST
    )
    try:
        data = load_path(path)
    except (OSError, ValueError) as exc:
        raise ValueError("quantum quality manifest is unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("quantum quality manifest must be a JSON object")
    for key in ("schema_version", "iteration", "n_total", "n_accepted", "n_rejected"):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("quantum quality " + key + " must be an exact integer")
    if data["schema_version"] != QUANTUM_QUALITY_SCHEMA_VERSION:
        raise ValueError("unsupported quantum quality manifest schema")
    if data.get("phase") != str(expected_phase):
        raise ValueError("quantum quality phase mismatch")
    if data["iteration"] != int(expected_iteration):
        raise ValueError("quantum quality iteration mismatch")
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("quantum quality records must be a list")
    expected_names = None if expected_pointdirs is None else {str(value) for value in expected_pointdirs}
    seen = set()
    accepted_count = 0
    canonical_method = (
        None if expected_method is None else canonicalise_aimall_method(expected_method)
    )
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("quantum quality record must be an object")
        name = record.get("pointdir")
        if not isinstance(name, str) or Path(name).name != name or not name:
            raise ValueError("quantum quality pointdir name is invalid")
        if name in seen:
            raise ValueError("duplicate quantum quality pointdir: " + name)
        seen.add(name)
        accepted = record.get("accepted")
        if not isinstance(accepted, bool):
            raise ValueError("quantum quality accepted must be a boolean")
        reasons = record.get("reasons")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError("quantum quality reasons must be non-empty strings")
        if accepted == bool(reasons):
            raise ValueError("quantum quality accepted/reasons are contradictory")
        if accepted:
            accepted_count += 1
            atom_count = record.get("atom_count")
            n_int = record.get("n_int")
            if (
                not isinstance(atom_count, int)
                or isinstance(atom_count, bool)
                or atom_count <= 0
                or not isinstance(n_int, int)
                or isinstance(n_int, bool)
                or n_int != atom_count
            ):
                raise ValueError("accepted quantum quality atom/INT counts are invalid")
            per_atom = record.get("per_atom")
            if not isinstance(per_atom, list) or len(per_atom) != atom_count:
                raise ValueError("accepted quantum quality per_atom count is invalid")
            expected_atom_names = record.get("expected_atom_names")
            if (
                not isinstance(expected_atom_names, list)
                or len(expected_atom_names) != atom_count
                or any(
                    not isinstance(name, str) or not name
                    for name in expected_atom_names
                )
                or len(expected_atom_names) != len(set(expected_atom_names))
            ):
                raise ValueError("accepted quantum quality atom ordering is invalid")
            atom_names = set()
            from ichor.core.common.constants import multipole_names

            for atom_index, atom in enumerate(per_atom):
                if not isinstance(atom, dict):
                    raise ValueError("quantum quality per_atom entry must be an object")
                atom_name = atom.get("atom")
                if not isinstance(atom_name, str) or not atom_name or atom_name in atom_names:
                    raise ValueError("quantum quality atom identity is invalid")
                if atom_name != expected_atom_names[atom_index]:
                    raise ValueError("quantum quality atom order does not match geometry")
                atom_names.add(atom_name)
                for numeric in ("iqa_ha", "integration_error"):
                    if _finite_float(atom.get(numeric)) is None:
                        raise ValueError("accepted quantum quality " + numeric + " is non-finite")
                if canonical_method is not None and atom.get("canonical_dft_model") != canonical_method:
                    raise ValueError("quantum quality DFT model mismatch")
                multipoles = atom.get("multipoles")
                if not isinstance(multipoles, dict) or set(multipoles) != set(
                    multipole_names
                ):
                    raise ValueError("quantum quality multipole key set is invalid")
                if any(_finite_float(value) is None for value in multipoles.values()):
                    raise ValueError("accepted quantum quality multipole is non-finite")
            for numeric in (
                "sum_iqa_ha",
                "wfn_total_energy_ha",
                "wfn_virial_ratio",
                "iqa_energy_recovery_error_ha",
                "max_abs_integration_error",
            ):
                if _finite_float(record.get(numeric)) is None:
                    raise ValueError("accepted quantum quality " + numeric + " is non-finite")
    if expected_names is not None and seen != expected_names:
        raise ValueError("quantum quality pointdir membership mismatch")
    if data["n_total"] != len(records):
        raise ValueError("quantum quality n_total mismatch")
    if data["n_accepted"] != accepted_count:
        raise ValueError("quantum quality n_accepted mismatch")
    if data["n_rejected"] != len(records) - accepted_count:
        raise ValueError("quantum quality n_rejected mismatch")
    return data


__all__ = [
    "QUANTUM_QUALITY_MANIFEST",
    "QUANTUM_QUALITY_SCHEMA_VERSION",
    "canonicalise_aimall_method",
    "evaluate_aimall_pointdir",
    "read_quantum_quality_manifest",
    "write_quantum_quality_manifest",
]
