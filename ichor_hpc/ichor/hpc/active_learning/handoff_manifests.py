"""Strict JSON contracts between ARIADNE, Phase B POLUS, and Gaussian.

These manifests are deliberately small and daemon-owned.  They record the
lineage that cannot be recovered safely from glob order once ARIADNE and
Phase B have filtered or reordered candidates.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .daemon.state import atomic_write_json


ARIADNE_RESULTS_FILENAME = "ARIADNE_RESULTS.json"
ARIADNE_RESULTS_SCHEMA_VERSION = 1
ARIADNE_LANDING_AUDIT_FILENAME = "ARIADNE_LANDING_AUDIT.json"
ARIADNE_LANDING_AUDIT_SCHEMA_VERSION = 1
PHASE_A_SAMPLE_FILENAME = "PHASE_A_SAMPLE.json"
PHASE_A_SAMPLE_SCHEMA_VERSION = 1
PHASE_B_SELECTION_FILENAME = "PHASE_B_SELECTION.json"
PHASE_B_SELECTION_SCHEMA_VERSION = 1
SEED_SELECTION_DIAGNOSTICS_FILENAME = "SEED_SELECTION_DIAGNOSTICS.json"
SEED_SELECTION_DIAGNOSTICS_SCHEMA_VERSION = 1
ACQUISITION_MATURITY_AUDIT_FILENAME = "ACQUISITION_MATURITY_AUDIT.json"
ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION = 1


class HandoffManifestError(ValueError):
    """Raised when a phase handoff manifest is missing or violates contract."""


def iteration_dir(campaign_dir: Any, iteration: int) -> Path:
    return (
        Path(campaign_dir)
        / "7_ACTIVE_LEARNING"
        / ("iteration-" + str(int(iteration)).zfill(4))
    )


def seeds_picked_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / "seeds_picked.json"


def seed_selection_diagnostics_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / SEED_SELECTION_DIAGNOSTICS_FILENAME


def acquisition_maturity_audit_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / ACQUISITION_MATURITY_AUDIT_FILENAME


def ariadne_results_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / ARIADNE_RESULTS_FILENAME


def ariadne_landing_audit_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / ARIADNE_LANDING_AUDIT_FILENAME


def phase_a_sample_manifest_path(initial_dir: Any) -> Path:
    return Path(initial_dir) / PHASE_A_SAMPLE_FILENAME


def phase_b_selection_path(iter_dir: Any) -> Path:
    return Path(iter_dir) / PHASE_B_SELECTION_FILENAME


def _finite_float(value: Any, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError("expected finite numeric value, got " + repr(value)) from exc
    if not math.isfinite(out):
        raise HandoffManifestError("expected finite numeric value, got " + repr(value))
    return out


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise HandoffManifestError("expected integer or null, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError("expected integer or null, got " + repr(value)) from exc


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise HandoffManifestError(label + " must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError(label + " must be an integer") from exc


def _json_number_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def resolve_handoff_path(
    root: Any,
    raw: Any,
    *,
    kind: str,
    must_exist: bool = True,
    directory: bool = False,
) -> Path:
    base = Path(root).resolve()
    p = Path(str(raw))
    if not p.is_absolute():
        p = base / p
    try:
        resolved = p.resolve(strict=False)
    except OSError as exc:
        raise HandoffManifestError(kind + " path cannot be resolved: " + str(p)) from exc
    if resolved != base and base not in resolved.parents:
        raise HandoffManifestError(kind + " path escapes handoff root: " + str(p))
    if p.is_symlink() or resolved.is_symlink():
        raise HandoffManifestError(kind + " path is a symlink: " + str(p))
    if must_exist:
        if directory:
            if not resolved.is_dir():
                raise FileNotFoundError(kind + " directory missing: " + str(resolved))
        elif not resolved.is_file():
            raise FileNotFoundError(kind + " file missing: " + str(resolved))
    return resolved


def load_seeds_picked(iter_dir: Any, *, expected_iteration: Optional[int] = None) -> Dict[str, Any]:
    """Read seeds_picked.json and return a normalised payload with seed_records.

    Older sidecars without seed_records are still normalised from the legacy
    frame_ids/indices fields so existing campaigns can be reconciled, but the
    returned value always exposes the strict per-seed record list.
    """
    path = seeds_picked_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("seeds_picked.json missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("seeds_picked.json unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("seeds_picked.json must be a JSON object: " + str(path))
    try:
        iteration = int(data.get("iteration"))
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError("seeds_picked.json iteration must be an integer") from exc
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError(
            "seeds_picked.json iteration mismatch: expected "
            + str(int(expected_iteration))
            + " got "
            + str(iteration)
        )
    frame_ids = data.get("frame_ids")
    if not isinstance(frame_ids, list):
        raise HandoffManifestError("seeds_picked.json frame_ids must be a list")
    frame_ids_norm = [_int_or_none(x) for x in frame_ids]
    n_picked = int(data.get("n_picked", len(frame_ids_norm)) or 0)
    if n_picked != len(frame_ids_norm):
        raise HandoffManifestError(
            "seeds_picked.json n_picked does not match frame_ids length"
        )
    indices = data.get("indices")
    if not isinstance(indices, list) or len(indices) != n_picked:
        indices = list(range(n_picked))
    indices_norm = [int(x) for x in indices]
    bulk = {int(x) for x in (data.get("bulk_indices") or [])}
    variance = {int(x) for x in (data.get("variance_indices") or [])}
    variances = data.get("variances") if isinstance(data.get("variances"), list) else []

    raw_records = data.get("seed_records")
    if raw_records is None:
        records = []
        for seed_index, frame_id in enumerate(frame_ids_norm):
            selection_index = int(indices_norm[seed_index])
            if selection_index in bulk:
                origin = "bulk"
            elif selection_index in variance:
                origin = "variance"
            else:
                origin = "unknown"
            variance_value = None
            if seed_index < len(variances) and variances[seed_index] is not None:
                variance_value = _finite_float(variances[seed_index], allow_none=True)
            records.append({
                "seed_index": int(seed_index),
                "frame_id": frame_id,
                "selection_index": selection_index,
                "selection_origin": origin,
                "variance_at_selection": variance_value,
            })
    else:
        if not isinstance(raw_records, list):
            raise HandoffManifestError("seed_records must be a list")
        records = []
        seen = set()
        for raw in raw_records:
            if not isinstance(raw, dict):
                raise HandoffManifestError("seed_records entries must be JSON objects")
            seed_index = int(raw.get("seed_index"))
            if not 0 <= seed_index < n_picked:
                raise HandoffManifestError("seed_index out of range: " + str(seed_index))
            if seed_index in seen:
                raise HandoffManifestError("duplicate seed_index in seed_records: " + str(seed_index))
            seen.add(seed_index)
            frame_id = _int_or_none(raw.get("frame_id"))
            if frame_id != frame_ids_norm[seed_index]:
                raise HandoffManifestError(
                    "seed_records frame_id mismatch for seed_index " + str(seed_index)
                )
            origin = str(raw.get("selection_origin", "unknown"))
            if origin not in ("bulk", "variance", "d_optimal", "unknown"):
                raise HandoffManifestError("unknown selection_origin: " + origin)
            record = {
                "seed_index": seed_index,
                "frame_id": frame_id,
                "selection_index": int(raw.get("selection_index", indices_norm[seed_index])),
                "selection_origin": origin,
                "variance_at_selection": _finite_float(
                    raw.get("variance_at_selection"), allow_none=True,
                ),
            }
            for key in (
                "raw_variance",
                "raw_score",
                "d_optimal_conditional_variance",
                "d_optimal_gain",
                "d_optimal_max_correlation_to_selected",
            ):
                if key in raw:
                    record[key] = _finite_float(raw.get(key), allow_none=True)
            for key in ("d_optimal_prefilter_rank", "variance_rank"):
                if key in raw and raw.get(key) is not None:
                    if isinstance(raw.get(key), bool):
                        raise HandoffManifestError(key + " must be an integer, got bool")
                    record[key] = int(raw.get(key))
            records.append(record)
        if len(seen) != n_picked:
            raise HandoffManifestError("seed_records does not cover every picked seed")
        records.sort(key=lambda rec: int(rec["seed_index"]))

    out = dict(data)
    out["iteration"] = iteration
    out["n_picked"] = n_picked
    out["frame_ids"] = frame_ids_norm
    out["seed_records"] = records
    return out


def write_seed_selection_diagnostics(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    data = dict(payload)
    data["schema_version"] = SEED_SELECTION_DIAGNOSTICS_SCHEMA_VERSION
    path = seed_selection_diagnostics_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return path


def read_seed_selection_diagnostics(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = seed_selection_diagnostics_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("seed selection diagnostics missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("seed selection diagnostics unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("seed selection diagnostics must be a JSON object")
    if _required_int(data.get("schema_version", -1), "seed selection diagnostics schema_version") != SEED_SELECTION_DIAGNOSTICS_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported seed selection diagnostics schema")
    iteration = _required_int(data.get("iteration"), "seed selection diagnostics iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("seed selection diagnostics iteration mismatch")
    selected = data.get("selected")
    if not isinstance(selected, list):
        raise HandoffManifestError("seed selection diagnostics selected must be a list")
    return data


def write_acquisition_maturity_audit(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    data = dict(payload)
    data["schema_version"] = ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION
    path = acquisition_maturity_audit_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return path


def acquisition_maturity_audit_payload(
    *,
    iteration: int,
    seed_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback_counts: Dict[str, int] = {}
    n_candidates = 0
    n_with_spectral = 0
    n_with_residual = 0
    n_with_banded_energy = 0
    seeds: List[Dict[str, Any]] = []
    for record in seed_records:
        candidates = []
        for candidate in list(record.get("landing_candidates") or []):
            if not isinstance(candidate, dict):
                continue
            metrics = candidate.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            n_candidates += 1
            if metrics.get("spectral_frequency_risk") is not None:
                n_with_spectral += 1
            if metrics.get("fullspace_residual_distance") is not None:
                n_with_residual += 1
            if metrics.get("banded_energy_risk") is not None:
                n_with_banded_energy += 1
            reasons = metrics.get("acquisition_fallback_reasons") or []
            if isinstance(reasons, list):
                for reason in reasons:
                    key = str(reason)
                    fallback_counts[key] = int(fallback_counts.get(key, 0)) + 1
            candidates.append({
                "candidate_index": candidate.get("candidate_index"),
                "origin": candidate.get("origin"),
                "accepted": bool(candidate.get("accepted", False)),
                "reasons": list(candidate.get("reasons") or []),
                "record_only_reasons": list(candidate.get("record_only_reasons") or []),
                "alpha": _json_number_or_none(candidate.get("alpha")),
                "informativeness_score": _json_number_or_none(candidate.get("informativeness_score")),
                "risk_penalty_score": _json_number_or_none(candidate.get("risk_penalty_score")),
                "metrics": {
                    key: (
                        list(metrics.get(key) or [])
                        if key == "acquisition_fallback_reasons"
                        else _json_number_or_none(metrics.get(key))
                    )
                    for key in (
                        "total_score",
                        "observable_score",
                        "outlier_penalty_score",
                        "energy_risk",
                        "banded_energy_risk",
                        "spectral_frequency_risk",
                        "legacy_frequency_risk",
                        "fullspace_residual_distance",
                        "fullspace_residual_penalty",
                        "aligned_rmsd_ang",
                        "aligned_rmsd_penalty",
                        "chemistry_penalty",
                        "distance_penalty",
                        "whitened_distance",
                        "acquisition_fallback_reasons",
                    )
                    if key in metrics
                },
            })
        selection_diagnostics = record.get("selection_diagnostics")
        if not isinstance(selection_diagnostics, dict):
            selection_diagnostics = None
        seeds.append({
            "seed_index": record.get("seed_index"),
            "result_json": record.get("result_json"),
            "landing_policy": (
                (record.get("landing_safety") or {}).get("policy")
                if isinstance(record.get("landing_safety"), dict)
                else None
            ),
            "landing_safety_metrics": (
                dict((record.get("landing_safety") or {}).get("metrics") or {})
                if isinstance(record.get("landing_safety"), dict)
                else {}
            ),
            "selection_diagnostics": selection_diagnostics,
            "landing_candidates": candidates,
        })
    return {
        "iteration": int(iteration),
        "summary": {
            "n_seeds": int(len(seed_records)),
            "n_candidates": int(n_candidates),
            "n_candidates_with_spectral": int(n_with_spectral),
            "n_candidates_with_fullspace_residual": int(n_with_residual),
            "n_candidates_with_banded_energy": int(n_with_banded_energy),
            "fallback_reasons": fallback_counts,
        },
        "seeds": seeds,
    }


def read_acquisition_maturity_audit(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = acquisition_maturity_audit_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("acquisition maturity audit missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("acquisition maturity audit unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("acquisition maturity audit must be a JSON object")
    if _required_int(data.get("schema_version", -1), "acquisition maturity audit schema_version") != ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported acquisition maturity audit schema")
    iteration = _required_int(data.get("iteration"), "acquisition maturity audit iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("acquisition maturity audit iteration mismatch")
    seeds = data.get("seeds")
    if not isinstance(seeds, list):
        raise HandoffManifestError("acquisition maturity audit seeds must be a list")
    return data


def validate_ariadne_result(
    result: Dict[str, Any],
    *,
    expected_iteration: int,
    seed_record: Dict[str, Any],
    expected_atom_types: Optional[Sequence[str]] = None,
    expected_trajectory_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HandoffManifestError("ARIADNE result must be a JSON object")
    seed_index = int(seed_record["seed_index"])
    frame_id = seed_record.get("frame_id")
    if int(result.get("iteration")) != int(expected_iteration):
        raise HandoffManifestError("wrong_iteration")
    if int(result.get("seed_index")) != seed_index:
        raise HandoffManifestError("wrong_seed_index")
    result_frame_id = _int_or_none(result.get("seed_frame_id"))
    if result_frame_id != frame_id:
        raise HandoffManifestError("wrong_seed_frame")
    if expected_trajectory_sha256:
        result_sha = result.get("trajectory_sha256")
        if result_sha is None:
            raise HandoffManifestError("missing_trajectory_sha256")
        if str(result_sha) != str(expected_trajectory_sha256):
            raise HandoffManifestError("wrong_trajectory_sha256")
    return_code = int(result.get("return_code"))
    if return_code != 0:
        try:
            from .acquisition.ariadne_runner import ariadne_result_usability_payload

            usability = ariadne_result_usability_payload(result)
        except Exception:
            usability = {"usable": False, "reason": "usability_check_failed"}
        if not bool(usability.get("usable", False)):
            raise HandoffManifestError(
                "ariadne_return_code_"
                + str(return_code)
                + ":"
                + str(usability.get("reason", "unusable_landing"))
            )
    alpha_initial = _finite_float(result.get("alpha_initial"))
    alpha_final = _finite_float(result.get("alpha_final"))
    n_evaluations = int(result.get("n_evaluations"))
    wall_seconds = _finite_float(result.get("wall_seconds"))
    atom_types = result.get("atom_types")
    final_coordinates = result.get("final_coordinates")
    if not isinstance(atom_types, list) or not atom_types:
        raise HandoffManifestError("missing_atom_types")
    clean_atom_types = [str(x) for x in atom_types]
    if expected_atom_types is not None:
        expected = [str(x) for x in expected_atom_types]
        if clean_atom_types != expected:
            raise HandoffManifestError("atom_type_order_mismatch")
    if not isinstance(final_coordinates, list) or len(final_coordinates) != len(atom_types):
        raise HandoffManifestError("final_coordinates_shape_mismatch")
    clean_coords = []
    for row in final_coordinates:
        if not isinstance(row, list) or len(row) != 3:
            raise HandoffManifestError("final_coordinates_shape_mismatch")
        clean_coords.append([_finite_float(row[0]), _finite_float(row[1]), _finite_float(row[2])])
    alpha_trajectory = result.get("alpha_trajectory")
    if not isinstance(alpha_trajectory, list):
        raise HandoffManifestError("alpha_trajectory_missing")
    for value in alpha_trajectory:
        _finite_float(value)
    whitened = _finite_float(result.get("whitened_distance_final"), allow_none=True)
    return {
        "seed_index": seed_index,
        "seed_frame_id": frame_id,
        "alpha_initial": alpha_initial,
        "alpha_final": alpha_final,
        "alpha_trajectory": [float(x) for x in alpha_trajectory],
        "n_evaluations": n_evaluations,
        "return_code": return_code,
        "trajectory_sha256": str(result.get("trajectory_sha256", "")),
        "wall_seconds": wall_seconds,
        "fell_back_to_ds": bool(result.get("fell_back_to_ds", False)),
        "whitened_distance_final": whitened,
        "atom_types": clean_atom_types,
        "final_coordinates": clean_coords,
    }


def write_ariadne_results_manifest(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    path = ariadne_results_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def write_ariadne_landing_audit(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    data = dict(payload)
    data["schema_version"] = ARIADNE_LANDING_AUDIT_SCHEMA_VERSION
    path = ariadne_landing_audit_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return path


def read_ariadne_landing_audit(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = ariadne_landing_audit_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("ARIADNE landing audit missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE landing audit unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("ARIADNE landing audit must be a JSON object")
    if _required_int(data.get("schema_version", -1), "ARIADNE landing audit schema_version") != ARIADNE_LANDING_AUDIT_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported ARIADNE landing audit schema")
    iteration = _required_int(data.get("iteration"), "ARIADNE landing audit iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("ARIADNE landing audit iteration mismatch")
    summary = data.get("summary")
    seeds = data.get("seeds")
    if not isinstance(summary, dict):
        raise HandoffManifestError("ARIADNE landing audit summary must be an object")
    if not isinstance(seeds, list):
        raise HandoffManifestError("ARIADNE landing audit seeds must be a list")
    seen = set()
    for rec in seeds:
        if not isinstance(rec, dict):
            raise HandoffManifestError("ARIADNE landing audit seed record must be an object")
        seed_index = _required_int(rec.get("seed_index"), "ARIADNE landing audit seed_index")
        if seed_index in seen:
            raise HandoffManifestError("duplicate ARIADNE landing audit seed_index")
        seen.add(seed_index)
    return data


def read_ariadne_results_manifest(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    require_nonempty: bool = True,
) -> Dict[str, Any]:
    path = ariadne_results_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("ARIADNE results manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE results manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("ARIADNE results manifest must be a JSON object")
    if int(data.get("schema_version", -1)) != ARIADNE_RESULTS_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported ARIADNE results manifest schema")
    iteration = int(data.get("iteration"))
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("ARIADNE results manifest iteration mismatch")
    accepted = data.get("accepted")
    rejected = data.get("rejected")
    if not isinstance(accepted, list):
        raise HandoffManifestError("ARIADNE results manifest accepted must be a list")
    if not isinstance(rejected, list):
        raise HandoffManifestError("ARIADNE results manifest rejected must be a list")
    n_accepted = data.get("n_accepted")
    if n_accepted is not None and int(n_accepted) != len(accepted):
        raise HandoffManifestError("ARIADNE n_accepted does not match accepted length")
    n_rejected = data.get("n_rejected")
    if n_rejected is not None and int(n_rejected) != len(rejected):
        raise HandoffManifestError("ARIADNE n_rejected does not match rejected length")
    expected_n = data.get("expected_n")
    if expected_n is not None and int(expected_n) != len(accepted) + len(rejected):
        raise HandoffManifestError("ARIADNE expected_n does not match accepted+rejected length")
    if require_nonempty and not accepted:
        raise HandoffManifestError("ARIADNE results manifest accepted list is empty")
    seen = set()
    normalised_accepted = []
    trajectory_sha = str(data.get("trajectory_sha256") or "")
    for rec in accepted:
        if not isinstance(rec, dict):
            raise HandoffManifestError("accepted ARIADNE record must be an object")
        seed_index = int(rec.get("seed_index"))
        if seed_index in seen:
            raise HandoffManifestError("duplicate accepted ARIADNE seed_index")
        seen.add(seed_index)
        out_rec = dict(rec)
        for key in ("seed_dir", "result_json", "provenance_json"):
            resolved = resolve_handoff_path(
                path.parent,
                rec.get(key, ""),
                kind="ARIADNE " + key,
                directory=(key == "seed_dir"),
            )
            out_rec[key] = str(resolved)
        try:
            result_payload = json.loads(Path(str(out_rec["result_json"])).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError("ARIADNE result unreadable: " + str(out_rec["result_json"])) from exc
        seed_record = {
            "seed_index": seed_index,
            "frame_id": _int_or_none(out_rec.get("seed_frame_id")),
        }
        validate_ariadne_result(
            result_payload,
            expected_iteration=iteration,
            seed_record=seed_record,
            expected_trajectory_sha256=trajectory_sha or None,
        )
        normalised_accepted.append(out_rec)
    normalised_rejected = []
    for rec in rejected:
        if not isinstance(rec, dict):
            raise HandoffManifestError("rejected ARIADNE record must be an object")
        out_rec = dict(rec)
        for key in ("seed_dir", "result_json", "provenance_json"):
            if key in out_rec and out_rec.get(key):
                resolved = resolve_handoff_path(
                    path.parent,
                    out_rec.get(key, ""),
                    kind="ARIADNE rejected " + key,
                    directory=(key == "seed_dir"),
                    must_exist=(key == "seed_dir"),
                )
                out_rec[key] = str(resolved)
        normalised_rejected.append(out_rec)
    out = dict(data)
    out["accepted"] = normalised_accepted
    out["rejected"] = normalised_rejected
    return out


def ariadne_candidate_frames(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    """Return accepted ARIADNE geometries as ICHOR Atoms plus manifest records."""
    from ichor.core.atoms import Atom, Atoms

    manifest = read_ariadne_results_manifest(
        iter_dir,
        expected_iteration=expected_iteration,
    )
    frames = []
    records = []
    for rec in manifest["accepted"]:
        result_path = Path(str(rec["result_json"]))
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError("ARIADNE result unreadable: " + str(result_path)) from exc
        seed_record = {
            "seed_index": int(rec["seed_index"]),
            "frame_id": _int_or_none(rec.get("seed_frame_id")),
        }
        validate_ariadne_result(
            result,
            expected_iteration=int(manifest["iteration"]),
            seed_record=seed_record,
            expected_trajectory_sha256=str(manifest.get("trajectory_sha256") or "") or None,
        )
        atom_types = result.get("atom_types") or []
        coords = result.get("final_coordinates") or []
        if not atom_types or len(atom_types) != len(coords):
            raise HandoffManifestError("ARIADNE result geometry shape mismatch: " + str(result_path))
        atoms = Atoms([
            Atom(str(t), _finite_float(c[0]), _finite_float(c[1]), _finite_float(c[2]))
            for t, c in zip(atom_types, coords)
        ])
        frames.append(atoms)
        records.append(dict(rec))
    return manifest, frames, records


def write_phase_a_sample_manifest(initial_dir: Any, payload: Dict[str, Any]) -> Path:
    path = phase_a_sample_manifest_path(initial_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = PHASE_A_SAMPLE_SCHEMA_VERSION
    atomic_write_json(path, data)
    return path


def read_phase_a_sample_manifest(
    initial_dir: Any,
    *,
    require_nonempty: bool = True,
) -> Dict[str, Any]:
    path = phase_a_sample_manifest_path(initial_dir)
    if not path.is_file():
        raise FileNotFoundError("Phase A sample manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("Phase A sample manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("Phase A sample manifest must be a JSON object")
    if int(data.get("schema_version", -1)) != PHASE_A_SAMPLE_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported Phase A sample manifest schema")
    if str(data.get("phase")) != "PHASE_A_POLUS":
        raise HandoffManifestError("Phase A sample manifest phase mismatch")
    if int(data.get("iteration")) != -1:
        raise HandoffManifestError("Phase A sample manifest iteration must be -1")
    try:
        n_select = int(data.get("n_select"))
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError("Phase A sample manifest n_select must be an integer") from exc
    if require_nonempty and n_select <= 0:
        raise HandoffManifestError("Phase A sample manifest n_select must be positive")
    root = Path(initial_dir)
    sample = resolve_handoff_path(
        root,
        data.get("sample_xyz", ""),
        kind="Phase A sample_xyz",
    )
    if not (sample.name.startswith("initial-SAMPLE-") and sample.suffix == ".xyz"):
        raise HandoffManifestError("Phase A sample path has unexpected name: " + str(sample))
    index_path = resolve_handoff_path(
        root,
        data.get("index_path", ""),
        kind="Phase A index_path",
    )
    if not (index_path.name.startswith("initial-INDEX-") and index_path.suffix == ".dat"):
        raise HandoffManifestError("Phase A index path has unexpected name: " + str(index_path))
    selected = data.get("selected_indices")
    if selected is not None:
        if not isinstance(selected, list):
            raise HandoffManifestError("Phase A selected_indices must be a list")
        if len(selected) != n_select:
            raise HandoffManifestError("Phase A selected_indices length mismatch")
        seen = set()
        for value in selected:
            try:
                idx = int(value)
            except (TypeError, ValueError) as exc:
                raise HandoffManifestError("Phase A selected_indices must be integers") from exc
            if idx in seen:
                raise HandoffManifestError("Phase A selected_indices contains duplicates")
            seen.add(idx)
    out = dict(data)
    out["n_select"] = n_select
    out["sample_xyz"] = str(sample)
    out["index_path"] = str(index_path)
    return out


def write_phase_b_selection_manifest(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    path = phase_b_selection_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def read_phase_b_selection_manifest(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    require_nonempty: bool = True,
) -> Dict[str, Any]:
    path = phase_b_selection_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("Phase B selection manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("Phase B selection manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("Phase B selection manifest must be a JSON object")
    if int(data.get("schema_version", -1)) != PHASE_B_SELECTION_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported Phase B selection manifest schema")
    iteration = int(data.get("iteration"))
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("Phase B selection manifest iteration mismatch")
    raw = data.get("raw")
    final = data.get("final")
    if not isinstance(raw, list):
        raise HandoffManifestError("Phase B selection manifest raw must be a list")
    if not isinstance(final, list):
        raise HandoffManifestError("Phase B selection manifest final must be a list")
    if data.get("n_selected_raw") is not None and int(data.get("n_selected_raw")) != len(raw):
        raise HandoffManifestError("Phase B n_selected_raw does not match raw length")
    if data.get("n_kept") is not None and int(data.get("n_kept")) != len(final):
        raise HandoffManifestError("Phase B n_kept does not match final length")
    source_manifest_raw = data.get("source_ariadne_manifest")
    source_manifest = None
    if source_manifest_raw:
        source_manifest = resolve_handoff_path(
            path.parent,
            source_manifest_raw,
            kind="Phase B source_ariadne_manifest",
        )
    if require_nonempty and not final:
        raise HandoffManifestError("Phase B selection manifest final list is empty")
    seen_raw = set()
    normalised_raw = []
    raw_kept_final_indexes = set()
    for raw_idx, rec in enumerate(raw):
        if not isinstance(rec, dict):
            raise HandoffManifestError("Phase B raw record must be an object")
        out_rec = dict(rec)
        declared_raw_index = int(out_rec.get("raw_index"))
        if declared_raw_index != raw_idx:
            raise HandoffManifestError("Phase B raw_index values must match raw order")
        if declared_raw_index in seen_raw:
            raise HandoffManifestError("duplicate Phase B raw_index")
        seen_raw.add(declared_raw_index)
        kept = bool(out_rec.get("kept_after_dedup", False))
        final_index_value = out_rec.get("final_index")
        if kept:
            if final_index_value is None:
                raise HandoffManifestError("Phase B kept raw record has null final_index")
            raw_kept_final_indexes.add(int(final_index_value))
        elif final_index_value is not None:
            raise HandoffManifestError("Phase B dropped raw record has non-null final_index")
        for key in ("seed_dir", "result_json", "provenance_json"):
            resolved = resolve_handoff_path(
                path.parent,
                out_rec.get(key, ""),
                kind="Phase B raw " + key,
                directory=(key == "seed_dir"),
            )
            out_rec[key] = str(resolved)
        normalised_raw.append(out_rec)
    seen_final = set()
    seen_final_raw = set()
    normalised_final = []
    for rec in final:
        if not isinstance(rec, dict):
            raise HandoffManifestError("Phase B final record must be an object")
        out_rec = dict(rec)
        final_index = int(rec.get("final_index"))
        if final_index in seen_final:
            raise HandoffManifestError("duplicate Phase B final_index")
        seen_final.add(final_index)
        raw_index = int(out_rec.get("raw_index"))
        if raw_index in seen_final_raw:
            raise HandoffManifestError("duplicate Phase B final raw_index")
        seen_final_raw.add(raw_index)
        if not bool(out_rec.get("kept_after_dedup", False)):
            raise HandoffManifestError("Phase B final record is not marked kept_after_dedup")
        for key in ("seed_dir", "result_json", "provenance_json"):
            resolved = resolve_handoff_path(
                path.parent,
                rec.get(key, ""),
                kind="Phase B " + key,
                directory=(key == "seed_dir"),
            )
            out_rec[key] = str(resolved)
        normalised_final.append(out_rec)
    if seen_final and sorted(seen_final) != list(range(len(seen_final))):
        raise HandoffManifestError("Phase B final_index values must be contiguous from zero")
    if seen_final != raw_kept_final_indexes:
        raise HandoffManifestError("Phase B final records do not match kept raw records")
    kept_raw_indexes = {
        int(rec["raw_index"]) for rec in normalised_raw if bool(rec.get("kept_after_dedup", False))
    }
    if seen_final_raw != kept_raw_indexes:
        raise HandoffManifestError("Phase B final raw_index set does not match kept raw records")
    out = dict(data)
    out["raw"] = normalised_raw
    out["final"] = normalised_final
    if source_manifest is not None:
        out["source_ariadne_manifest"] = str(source_manifest)
    return out
