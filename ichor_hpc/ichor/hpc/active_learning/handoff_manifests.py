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
            if origin not in ("bulk", "variance", "unknown"):
                raise HandoffManifestError("unknown selection_origin: " + origin)
            records.append({
                "seed_index": seed_index,
                "frame_id": frame_id,
                "selection_index": int(raw.get("selection_index", indices_norm[seed_index])),
                "selection_origin": origin,
                "variance_at_selection": _finite_float(
                    raw.get("variance_at_selection"), allow_none=True,
                ),
            })
        if len(seen) != n_picked:
            raise HandoffManifestError("seed_records does not cover every picked seed")
        records.sort(key=lambda rec: int(rec["seed_index"]))

    out = dict(data)
    out["iteration"] = iteration
    out["n_picked"] = n_picked
    out["frame_ids"] = frame_ids_norm
    out["seed_records"] = records
    return out


def validate_ariadne_result(
    result: Dict[str, Any],
    *,
    expected_iteration: int,
    seed_record: Dict[str, Any],
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
    return_code = int(result.get("return_code"))
    if return_code != 0:
        raise HandoffManifestError("ariadne_return_code_" + str(return_code))
    alpha_initial = _finite_float(result.get("alpha_initial"))
    alpha_final = _finite_float(result.get("alpha_final"))
    n_evaluations = int(result.get("n_evaluations"))
    wall_seconds = _finite_float(result.get("wall_seconds"))
    atom_types = result.get("atom_types")
    final_coordinates = result.get("final_coordinates")
    if not isinstance(atom_types, list) or not atom_types:
        raise HandoffManifestError("missing_atom_types")
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
        "wall_seconds": wall_seconds,
        "fell_back_to_ds": bool(result.get("fell_back_to_ds", False)),
        "whitened_distance_final": whitened,
        "atom_types": [str(x) for x in atom_types],
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
    if int(data.get("schema_version", -1)) != ARIADNE_LANDING_AUDIT_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported ARIADNE landing audit schema")
    iteration = int(data.get("iteration"))
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
        seed_index = int(rec.get("seed_index"))
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
    if require_nonempty and not accepted:
        raise HandoffManifestError("ARIADNE results manifest accepted list is empty")
    seen = set()
    for rec in accepted:
        if not isinstance(rec, dict):
            raise HandoffManifestError("accepted ARIADNE record must be an object")
        seed_index = int(rec.get("seed_index"))
        if seed_index in seen:
            raise HandoffManifestError("duplicate accepted ARIADNE seed_index")
        seen.add(seed_index)
        for key in ("seed_dir", "result_json", "provenance_json"):
            p = Path(str(rec.get(key, "")))
            if not p.is_file() and key != "seed_dir":
                raise FileNotFoundError("ARIADNE manifest path missing: " + str(p))
            if key == "seed_dir" and not p.is_dir():
                raise FileNotFoundError("ARIADNE seed_dir missing: " + str(p))
    return data


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
    sample = Path(str(data.get("sample_xyz", "")))
    if not sample.is_absolute():
        sample = root / sample
    if not sample.is_file():
        raise FileNotFoundError("Phase A sample file missing: " + str(sample))
    if not (sample.name.startswith("initial-SAMPLE-") and sample.suffix == ".xyz"):
        raise HandoffManifestError("Phase A sample path has unexpected name: " + str(sample))
    index_path = Path(str(data.get("index_path", "")))
    if not index_path.is_absolute():
        index_path = root / index_path
    if not index_path.is_file():
        raise FileNotFoundError("Phase A index file missing: " + str(index_path))
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
    if require_nonempty and not final:
        raise HandoffManifestError("Phase B selection manifest final list is empty")
    seen_final = set()
    for rec in final:
        if not isinstance(rec, dict):
            raise HandoffManifestError("Phase B final record must be an object")
        final_index = int(rec.get("final_index"))
        if final_index in seen_final:
            raise HandoffManifestError("duplicate Phase B final_index")
        seen_final.add(final_index)
        for key in ("seed_dir", "result_json", "provenance_json"):
            p = Path(str(rec.get(key, "")))
            if key == "seed_dir":
                if not p.is_dir():
                    raise FileNotFoundError("Phase B seed_dir missing: " + str(p))
            elif not p.is_file():
                raise FileNotFoundError("Phase B source path missing: " + str(p))
    if seen_final and sorted(seen_final) != list(range(len(seen_final))):
        raise HandoffManifestError("Phase B final_index values must be contiguous from zero")
    return data
