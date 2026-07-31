"""Derived, authority-bound history used by adaptive sampling.

The cache in this module is deliberately non-authoritative.  A cold build
uses the existing strict ARIADNE handoff reader; a warm read is admitted only
when the immutable control manifests and their task-map binding are unchanged.
"""

from __future__ import annotations

from .strict_json import strict_json as json
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import uuid

from .daemon.filesystem import campaign_owned_path
from .daemon.state import atomic_write_json
from .handoff_manifests import (
    ARIADNE_LANDING_AUDIT_SCHEMA_VERSION,
    ARIADNE_RESULTS_SCHEMA_VERSION,
    HandoffManifestError,
    ariadne_landing_audit_path,
    ariadne_results_path,
    ariadne_task_map_path,
)
from .historical_ariadne import read_historical_ariadne_records
from .versioning.manifest import sha256_file


SAMPLING_HISTORY_CACHE_SCHEMA_VERSION = 1
SAMPLING_HISTORY_ENCODING_VERSION = 1
SAMPLING_HISTORY_MAX_READERS = 8


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_regular_json(path: Path, label: str) -> Dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HandoffManifestError(label + " is missing or unsafe: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError(label + " is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise HandoffManifestError(label + " must contain a JSON object")
    return payload


def _sampling_protocol_identity(iter_dir: Path) -> Dict[str, Any]:
    from .sampling_protocol import sampling_protocol_resolved_path

    path = sampling_protocol_resolved_path(iter_dir)
    if not path.exists() and not path.is_symlink():
        return {"status": "missing", "sha256": None}
    if path.is_symlink() or not path.is_file():
        return {"status": "invalid", "sha256": None}
    return {"status": "present", "sha256": sha256_file(path)}


def _normalised_symbol_source(symbol: Any) -> str:
    tree = ast.parse(inspect.getsource(symbol))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:]
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def _producer_fingerprint() -> str:
    payload = {
        "encoding_version": SAMPLING_HISTORY_ENCODING_VERSION,
        "symbols": [
            _normalised_symbol_source(_coordinates),
            _normalised_symbol_source(per_atom_displacements),
            _normalised_symbol_source(_build_records),
        ],
    }
    return _canonical_sha256(payload)


def _required_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise HandoffManifestError(label + " must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError(label + " must be an integer") from exc
    if out < int(minimum):
        raise HandoffManifestError(label + " is outside its valid range")
    return out


def _control_identity(iter_dir: Path, iteration: int) -> Dict[str, Any]:
    results_path = ariadne_results_path(iter_dir)
    audit_path = ariadne_landing_audit_path(iter_dir)
    task_map_path = ariadne_task_map_path(iter_dir)
    results = _read_regular_json(results_path, "ARIADNE results manifest")
    audit = _read_regular_json(audit_path, "ARIADNE landing audit")
    if _required_int(results.get("schema_version"), "results schema") != int(
        ARIADNE_RESULTS_SCHEMA_VERSION
    ):
        raise HandoffManifestError("unsupported ARIADNE results schema")
    if _required_int(audit.get("schema_version"), "audit schema") != int(
        ARIADNE_LANDING_AUDIT_SCHEMA_VERSION
    ):
        raise HandoffManifestError("unsupported ARIADNE landing audit schema")
    if _required_int(results.get("iteration"), "results iteration") != int(iteration):
        raise HandoffManifestError("ARIADNE results iteration mismatch")
    if _required_int(audit.get("iteration"), "audit iteration") != int(iteration):
        raise HandoffManifestError("ARIADNE audit iteration mismatch")
    accepted = results.get("accepted")
    rejected = results.get("rejected")
    audit_records = audit.get("seeds")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise HandoffManifestError("ARIADNE result dispositions are invalid")
    if not isinstance(audit_records, list):
        raise HandoffManifestError("ARIADNE landing audit records are invalid")
    expected_n = _required_int(results.get("expected_n"), "results expected_n")
    if expected_n != len(accepted) + len(rejected):
        raise HandoffManifestError("ARIADNE results count mismatch")
    if _required_int(results.get("n_accepted"), "results n_accepted") != len(
        accepted
    ):
        raise HandoffManifestError("ARIADNE accepted count mismatch")
    if _required_int(results.get("n_rejected"), "results n_rejected") != len(
        rejected
    ):
        raise HandoffManifestError("ARIADNE rejected count mismatch")
    binding = results.get("task_map")
    if not isinstance(binding, dict):
        raise HandoffManifestError("ARIADNE task-map binding is missing")
    if task_map_path.is_symlink() or not task_map_path.is_file():
        raise HandoffManifestError("ARIADNE task map is missing or unsafe")
    if str(binding.get("sha256") or "") != sha256_file(task_map_path):
        raise HandoffManifestError("ARIADNE task-map hash mismatch")

    result_identities: List[Dict[str, Any]] = []
    all_seed_ids = set()
    for disposition, records in (("accepted", accepted), ("rejected", rejected)):
        for record in records:
            if not isinstance(record, dict):
                raise HandoffManifestError("ARIADNE result record is invalid")
            seed_id = _required_int(record.get("seed_id"), "result seed_id", minimum=1)
            if seed_id in all_seed_ids:
                raise HandoffManifestError("duplicate ARIADNE result seed_id")
            all_seed_ids.add(seed_id)
            result_identities.append(
                {
                    "disposition": disposition,
                    "seed_id": seed_id,
                    "seed_uid": str(record.get("seed_uid") or ""),
                    "result_json": str(record.get("result_json") or ""),
                    "result_sha256": str(record.get("result_sha256") or ""),
                    "output_manifest_sha256": str(
                        record.get("output_manifest_sha256") or ""
                    ),
                }
            )
    audit_ids = []
    for record in audit_records:
        if not isinstance(record, dict):
            raise HandoffManifestError("ARIADNE audit record is invalid")
        audit_ids.append(
            {
                "seed_id": _required_int(
                    record.get("seed_id"), "audit seed_id", minimum=1
                ),
                "seed_uid": str(record.get("seed_uid") or ""),
                "handoff_accepted": bool(record.get("handoff_accepted", False)),
            }
        )
    if {item["seed_id"] for item in audit_ids} != all_seed_ids:
        raise HandoffManifestError("ARIADNE audit/results seed membership mismatch")
    accepted_ids = {
        item["seed_id"] for item in result_identities if item["disposition"] == "accepted"
    }
    audit_by_id = {item["seed_id"]: item for item in audit_ids}
    for item in result_identities:
        audit_item = audit_by_id[item["seed_id"]]
        if item["seed_uid"] != audit_item["seed_uid"]:
            raise HandoffManifestError("ARIADNE audit/results seed UID mismatch")
        if item["seed_id"] in accepted_ids and not audit_item["handoff_accepted"]:
            raise HandoffManifestError("accepted ARIADNE result is rejected by its audit")
    return {
        "campaign_uid": str(results.get("campaign_uid") or ""),
        "iteration": int(iteration),
        "results_sha256": sha256_file(results_path),
        "landing_audit_sha256": sha256_file(audit_path),
        "task_map_sha256": sha256_file(task_map_path),
        "result_identities": result_identities,
        "sampling_protocol": _sampling_protocol_identity(iter_dir),
        "producer_fingerprint_sha256": _producer_fingerprint(),
    }


def _coordinates(payload: Mapping[str, Any], *names: str) -> Optional[List[List[float]]]:
    for name in names:
        raw = payload.get(name)
        if not isinstance(raw, list):
            continue
        rows: List[List[float]] = []
        valid = True
        for row in raw:
            if not isinstance(row, list) or len(row) != 3:
                valid = False
                break
            values: List[float] = []
            for value in row:
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    valid = False
                    break
                if not math.isfinite(number):
                    valid = False
                    break
                values.append(number)
            if not valid:
                break
            rows.append(values)
        if valid and rows:
            return rows
    return None


def per_atom_displacements(result: Mapping[str, Any]) -> List[float]:
    """Return the legacy aligned per-atom displacement vector."""
    start = _coordinates(result, "seed_coordinates", "initial_coordinates")
    final = _coordinates(result, "final_coordinates")
    if start is None or final is None or len(start) != len(final):
        return []
    atom_types = result.get("atom_types")
    if (
        not isinstance(atom_types, list)
        or len(atom_types) != len(start)
        or any(not isinstance(symbol, str) or not symbol for symbol in atom_types)
    ):
        return []
    try:
        from ichor.core.adversarial.geometry import aligned_per_atom_displacements
        from ichor.core.atoms import Atom, Atoms

        reference = Atoms(
            [
                Atom(str(symbol), float(coords[0]), float(coords[1]), float(coords[2]))
                for symbol, coords in zip(atom_types, start)
            ]
        )
        mobile = Atoms(
            [
                Atom(str(symbol), float(coords[0]), float(coords[1]), float(coords[2]))
                for symbol, coords in zip(atom_types, final)
            ]
        )
        values = aligned_per_atom_displacements(reference, mobile)
    except Exception:
        return []
    return [
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) >= 0.0
    ]


def _record_metrics(record: Mapping[str, Any]) -> Dict[str, float]:
    safety = record.get("landing_safety")
    raw = safety.get("metrics") if isinstance(safety, dict) else None
    if not isinstance(raw, dict):
        raw = record.get("metrics")
    if not isinstance(raw, dict):
        return {}
    metrics: Dict[str, float] = {}
    for key, value in raw.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            metrics[str(key)] = number
    return metrics


def _load_result(record: Mapping[str, Any]) -> Tuple[bool, List[float]]:
    path = Path(str(record.get("result_json") or ""))
    try:
        payload = _read_regular_json(path, "historical ARIADNE result")
    except HandoffManifestError:
        return False, []
    return True, per_atom_displacements(payload)


def _build_records(iter_dir: Path, iteration: int) -> Dict[str, Any]:
    records = read_historical_ariadne_records(
        iter_dir,
        expected_iteration=int(iteration),
    )
    workers = max(1, min(SAMPLING_HISTORY_MAX_READERS, len(records) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        result_values = list(executor.map(_load_result, records))
    output = []
    for record, (result_read, displacements) in zip(records, result_values):
        output.append(
            {
                "seed_id": int(record["seed_id"]),
                "seed_uid": str(record.get("seed_uid") or ""),
                "metrics": _record_metrics(record),
                "result_json_read": bool(result_read),
                "per_atom_displacements": [float(value) for value in displacements],
            }
        )
    return {
        "schema_version": SAMPLING_HISTORY_CACHE_SCHEMA_VERSION,
        "iteration": int(iteration),
        "records": output,
    }


def _validate_records(
    payload: Mapping[str, Any],
    *,
    iteration: int,
    identity: Mapping[str, Any],
) -> Dict[str, Any]:
    if payload.get("schema_version") != SAMPLING_HISTORY_CACHE_SCHEMA_VERSION:
        raise ValueError("sampling-history cache schema is unsupported")
    if int(payload.get("iteration", -1)) != int(iteration):
        raise ValueError("sampling-history cache iteration mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("sampling-history cache records are invalid")
    accepted_identity = [
        item
        for item in identity["result_identities"]
        if item["disposition"] == "accepted"
    ]
    if len(records) != len(accepted_identity):
        raise ValueError("sampling-history cache record count mismatch")
    normalised = []
    for record, expected in zip(records, accepted_identity):
        if not isinstance(record, dict):
            raise ValueError("sampling-history cache record is invalid")
        if int(record.get("seed_id", -1)) != int(expected["seed_id"]):
            raise ValueError("sampling-history cache seed order mismatch")
        if str(record.get("seed_uid") or "") != str(expected["seed_uid"]):
            raise ValueError("sampling-history cache seed identity mismatch")
        metrics = record.get("metrics")
        displacements = record.get("per_atom_displacements")
        if not isinstance(metrics, dict) or not isinstance(displacements, list):
            raise ValueError("sampling-history cache numeric payload is invalid")
        clean_metrics = {}
        for key, value in metrics.items():
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("sampling-history cache metric is non-finite")
            clean_metrics[str(key)] = number
        clean_displacements = []
        for value in displacements:
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("sampling-history cache displacement is invalid")
            clean_displacements.append(number)
        normalised.append(
            {
                "seed_id": int(expected["seed_id"]),
                "seed_uid": str(expected["seed_uid"]),
                "metrics": clean_metrics,
                "result_json_read": bool(record.get("result_json_read", False)),
                "per_atom_displacements": clean_displacements,
            }
        )
    return {
        "schema_version": SAMPLING_HISTORY_CACHE_SCHEMA_VERSION,
        "iteration": int(iteration),
        "records": normalised,
    }


def _cache_paths(campaign_dir: Path, iteration: int, cache_id: str) -> Tuple[Path, Path, Path]:
    cache_token = str(cache_id)[:32]
    root = campaign_owned_path(
        campaign_dir,
        Path(".DATA")
        / "CACHE"
        / "SAMPLING_HISTORY"
        / ("iteration-" + str(int(iteration)).zfill(6))
        / cache_token,
    )
    return root, root / "records.json", root / "CACHE_MANIFEST.json"


def _read_cache(
    records_path: Path,
    manifest_path: Path,
    *,
    identity: Mapping[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    if (
        records_path.is_symlink()
        or manifest_path.is_symlink()
        or not records_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ValueError("sampling-history cache files are missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SAMPLING_HISTORY_CACHE_SCHEMA_VERSION:
        raise ValueError("sampling-history cache manifest schema is unsupported")
    if manifest.get("identity") != dict(identity):
        raise ValueError("sampling-history cache identity mismatch")
    binding = manifest.get("records")
    if not isinstance(binding, dict):
        raise ValueError("sampling-history cache record binding is invalid")
    if int(binding.get("size", -1)) != int(records_path.stat().st_size):
        raise ValueError("sampling-history cache size mismatch")
    if str(binding.get("sha256") or "") != sha256_file(records_path):
        raise ValueError("sampling-history cache hash mismatch")
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("sampling-history cache payload is invalid")
    return _validate_records(payload, iteration=iteration, identity=identity)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _publish_cache(
    root: Path,
    records_path: Path,
    manifest_path: Path,
    *,
    identity: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / (".building-" + root.name + "-" + uuid.uuid4().hex[:12])
    temporary.mkdir()
    retired: Optional[Path] = None
    published = False
    try:
        temporary_records = temporary / records_path.name
        temporary_manifest = temporary / manifest_path.name
        atomic_write_json(temporary_records, dict(payload))
        atomic_write_json(
            temporary_manifest,
            {
                "schema_version": SAMPLING_HISTORY_CACHE_SCHEMA_VERSION,
                "identity": dict(identity),
                "records": {
                    "size": int(temporary_records.stat().st_size),
                    "sha256": sha256_file(temporary_records),
                },
                "generated_at_iso": datetime.now(timezone.utc).isoformat(),
            },
        )
        if root.exists() or root.is_symlink():
            if root.is_symlink() or not root.is_dir():
                raise ValueError("sampling-history cache destination is unsafe")
            retired = parent / (".retired-" + uuid.uuid4().hex[:12])
            os.replace(root, retired)
            _fsync_directory(parent)
        os.replace(temporary, root)
        published = True
        _fsync_directory(parent)
        if retired is not None:
            shutil.rmtree(retired, ignore_errors=True)
    finally:
        if not published and retired is not None and retired.exists():
            if not root.exists() and not root.is_symlink():
                try:
                    os.replace(retired, root)
                    _fsync_directory(parent)
                except OSError:
                    pass
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary, ignore_errors=True)


def load_or_build_sampling_history(
    campaign_dir: Path,
    *,
    iteration: int,
    write_cache: bool = True,
) -> Dict[str, Any]:
    """Return ordered raw history, rebuilding derived evidence when necessary."""
    import portalocker

    campaign = Path(campaign_dir).resolve()
    from .layout import active_iteration_dir

    iter_dir = active_iteration_dir(campaign, int(iteration))
    identity = _control_identity(iter_dir, int(iteration))
    cache_id = _canonical_sha256(identity)

    def _strict_payload() -> Dict[str, Any]:
        payload = _build_records(iter_dir, int(iteration))
        return _validate_records(
            payload,
            iteration=int(iteration),
            identity=identity,
        )

    try:
        root, records_path, manifest_path = _cache_paths(
            campaign, int(iteration), cache_id
        )
    except (OSError, ValueError):
        return _strict_payload()
    if not bool(write_cache):
        try:
            return _read_cache(
                records_path,
                manifest_path,
                identity=identity,
                iteration=int(iteration),
            )
        except Exception:
            return _strict_payload()
    try:
        root.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError):
        return _strict_payload()
    lock_path = root.parent / (root.name + ".lock")
    try:
        with portalocker.Lock(str(lock_path), mode="a", timeout=120):
            try:
                return _read_cache(
                    records_path,
                    manifest_path,
                    identity=identity,
                    iteration=int(iteration),
                )
            except Exception:
                payload = _build_records(iter_dir, int(iteration))
                payload = _validate_records(
                    payload,
                    iteration=int(iteration),
                    identity=identity,
                )
                try:
                    _publish_cache(
                        root,
                        records_path,
                        manifest_path,
                        identity=identity,
                        payload=payload,
                    )
                    for sibling in root.parent.iterdir():
                        if (
                            sibling != root
                            and sibling.is_dir()
                            and not sibling.is_symlink()
                            and not sibling.name.startswith(".building-")
                        ):
                            shutil.rmtree(sibling, ignore_errors=True)
                except Exception:
                    pass
                return payload
    except portalocker.exceptions.LockException:
        return _strict_payload()


def prewarm_sampling_history_cache(iter_dir: Path, *, iteration: int) -> None:
    """Best-effort prewarm hook used after an ARIADNE publication commits."""
    root = Path(iter_dir).resolve()
    campaign = root.parent.parent
    load_or_build_sampling_history(campaign, iteration=int(iteration))


__all__ = [
    "SAMPLING_HISTORY_CACHE_SCHEMA_VERSION",
    "load_or_build_sampling_history",
    "per_atom_displacements",
    "prewarm_sampling_history_cache",
]
