"""Immutable FEREBUS feature contracts, per-point shards and version caches."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import ALF
from ichor.core.calculators import calculate_alf_atom_sequence
from ichor.core.calculators.features.alf_features_calculator import (
    calculate_alf_features,
)
from ichor.core.files import PointDirectory

from ..layout import active_learning_dir
from ..strict_json import load_path
from ..versioning.manifest import sha256_file
from ..versioning.provenance import read_provenance
from ..versioning.reference_data import canonical_json_sha256
from .filesystem import campaign_owned_path
from .state import _fsync_parent_dir, atomic_write_json


FEREBUS_FEATURE_CONTRACT = "FEREBUS_FEATURE_CONTRACT.json"
FEREBUS_FEATURE_CONTRACT_SCHEMA_VERSION = 1
FEREBUS_ROW_SHARD = "ROW_SHARD.json"
FEREBUS_ROW_SHARD_SCHEMA_VERSION = 1
FEREBUS_ROW_SHARD_ARRAY = "ROWS.npy"
FEREBUS_ROW_CACHE = "ROW_CACHE.json"
FEREBUS_ROW_CACHE_SCHEMA_VERSION = 1
FEREBUS_ROW_ENCODING_VERSION = 1
ROW_CACHE_PROGRESS_INTERVAL = 32

RowProgressCallback = Optional[
    Callable[[str, Mapping[str, Any]], None]
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def feature_contract_path(campaign_dir: Path) -> Path:
    return active_learning_dir(campaign_dir) / FEREBUS_FEATURE_CONTRACT


def row_cache_root(campaign_dir: Path, contract_sha256: str) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "CACHE"
        / "FEREBUS_ROWS"
        / str(contract_sha256)
    )


def row_cache_path(campaign_dir: Path, contract_sha256: str, version: int) -> Path:
    return row_cache_root(campaign_dir, contract_sha256) / (
        "iteration-" + str(int(version)).zfill(6)
    )


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # The final cache path already contains a full contract digest.  Keep the
    # temporary leaf compact so atomic writes also work on legacy Windows test
    # hosts with the traditional 260-character path limit.
    temporary = target.with_name(".n-" + uuid.uuid4().hex[:8])
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array, dtype=np.float64), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        _fsync_parent_dir(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_derived_tree(path: Path) -> None:
    """Remove an incomplete derived tree only after rejecting symlinks."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("derived FEREBUS cache path is not a regular directory")
    for child in root.rglob("*"):
        if child.is_symlink():
            raise ValueError(
                "derived FEREBUS cache contains a symlink: " + str(child)
            )
    shutil.rmtree(root)


def _normalise_alf(system_alf: Mapping[str, Any]) -> Dict[str, list[Optional[int]]]:
    payload: Dict[str, list[Optional[int]]] = {}
    for atom, alf in system_alf.items():
        values = [None if value is None else int(value) for value in alf]
        if len(values) != 3 or values[2] is None:
            raise ValueError(
                "FEREBUS feature contract requires the native three-index ALF for "
                + str(atom)
            )
        if any(value is None or int(value) < 0 for value in values):
            raise ValueError("FEREBUS feature contract contains an invalid ALF")
        payload[str(atom)] = values
    return payload


def _contract_system_alf(contract: Mapping[str, Any]) -> Dict[str, ALF]:
    raw = contract.get("alfs_zero_based")
    if not isinstance(raw, dict):
        raise ValueError("FEREBUS feature contract ALFs are missing")
    result: Dict[str, ALF] = {}
    for atom in contract["atom_names"]:
        values = raw.get(atom)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError("FEREBUS feature contract ALF is invalid for " + str(atom))
        result[str(atom)] = ALF(*(int(value) for value in values))
    return result


def system_alf_from_contract(contract: Mapping[str, Any]) -> Dict[str, ALF]:
    """Return the exact zero-based ALFs frozen by a feature contract."""
    return _contract_system_alf(contract)


def _build_contract(campaign: Path, config: Any, pointdir: Path) -> Dict[str, Any]:
    point = PointDirectory(pointdir)
    atom_names = [str(value) for value in point.atoms.atom_names]
    if len(atom_names) < 3:
        raise ValueError(
            "FEREBUS training requires at least three atoms for its native ALF ABI"
        )
    properties = [str(value) for value in config.ferebus.properties]
    if not properties:
        raise ValueError("ferebus.properties must not be empty")

    from . import input_staging as _staging

    bootstrap = _staging._model_bootstrap_context(campaign)
    if bootstrap is None:
        system_alf = point.atoms.alf_dict(calculate_alf_atom_sequence)
        alf_source = {
            "mode": "sequence",
            "calculator": "calculate_alf_atom_sequence",
        }
    else:
        tasks = _staging._load_model_bootstrap_tasks(bootstrap)
        system_alf = _staging._model_bootstrap_system_alf(
            bootstrap,
            tasks,
            properties,
        )
        alf_source = {
            "mode": "model_bootstrap",
            "manifest_sha256": sha256_file(bootstrap["manifest_path"]),
        }
    if list(system_alf) != atom_names:
        raise ValueError("FEREBUS ALF atom order differs from the staged geometry")
    features = np.asarray(
        point.features(calculate_alf_features, system_alf),
        dtype=np.float64,
    )
    if (
        features.ndim != 2
        or features.shape[0] != len(atom_names)
        or features.shape[1] <= 0
        or not np.isfinite(features).all()
    ):
        raise ValueError("FEREBUS feature contract probe produced invalid features")
    provenance = read_provenance(pointdir)
    base = {
        "schema_version": FEREBUS_FEATURE_CONTRACT_SCHEMA_VERSION,
        "row_encoding_version": FEREBUS_ROW_ENCODING_VERSION,
        "campaign_uid": str(provenance["campaign_uid"]),
        "atom_names": atom_names,
        "elements": [str(atom.type) for atom in point.atoms],
        "alfs_zero_based": _normalise_alf(system_alf),
        "alf_source": alf_source,
        "properties": properties,
        "feature_headers": ["f" + str(index + 1) for index in range(features.shape[1])],
        "row_headers": [
            "f" + str(index + 1) for index in range(features.shape[1])
        ]
        + properties,
        "dtype": "float64",
    }
    return {**base, "contract_sha256": canonical_json_sha256(base)}


def read_feature_contract(path_or_campaign: Path) -> Dict[str, Any]:
    candidate = Path(path_or_campaign)
    path = candidate if candidate.name == FEREBUS_FEATURE_CONTRACT else feature_contract_path(candidate)
    try:
        payload = load_path(path)
    except (OSError, ValueError) as exc:
        raise ValueError("FEREBUS feature contract is unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError("FEREBUS feature contract must be a JSON object")
    if payload.get("schema_version") != FEREBUS_FEATURE_CONTRACT_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS feature-contract schema")
    row_encoding_version = payload.get("row_encoding_version")
    if (
        isinstance(row_encoding_version, bool)
        or not isinstance(row_encoding_version, int)
        or row_encoding_version <= 0
    ):
        raise ValueError("FEREBUS feature-contract row encoding is invalid")
    digest = str(payload.get("contract_sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("contract_sha256", None)
    if digest != canonical_json_sha256(unsigned):
        raise ValueError("FEREBUS feature-contract digest mismatch")
    atom_names = payload.get("atom_names")
    properties = payload.get("properties")
    headers = payload.get("row_headers")
    if (
        not isinstance(atom_names, list)
        or len(atom_names) < 3
        or len(atom_names) != len(set(atom_names))
        or not isinstance(properties, list)
        or not properties
        or not isinstance(headers, list)
        or len(headers) <= len(properties)
    ):
        raise ValueError("FEREBUS feature-contract dimensions are invalid")
    _contract_system_alf(payload)
    return payload


def _preserve_feature_contract(campaign: Path, contract: Mapping[str, Any]) -> None:
    contract_sha = str(contract["contract_sha256"])
    historical_path = row_cache_root(
        campaign,
        contract_sha,
    ) / FEREBUS_FEATURE_CONTRACT
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    if historical_path.exists():
        historical = read_feature_contract(historical_path)
        if historical != dict(contract):
            raise ValueError(
                "preserved FEREBUS feature contract conflicts with the active contract"
            )
        return
    atomic_write_json(historical_path, dict(contract))


def ensure_current_feature_contract(campaign_dir: Path) -> Dict[str, Any]:
    """Advance the active contract when row semantics have been versioned.

    Existing cache namespaces are deliberately retained.  The updated contract
    digest selects a new namespace which is populated lazily by the scientific
    consumer.
    """
    campaign = Path(campaign_dir).resolve()
    path = feature_contract_path(campaign)
    observed = read_feature_contract(path)
    observed_version = int(observed["row_encoding_version"])
    if observed_version == FEREBUS_ROW_ENCODING_VERSION:
        return observed
    if observed_version > FEREBUS_ROW_ENCODING_VERSION:
        raise ValueError(
            "FEREBUS feature contract uses a newer row encoding than this ICHOR "
            "installation"
        )

    _preserve_feature_contract(campaign, observed)

    updated = dict(observed)
    updated.pop("contract_sha256", None)
    updated["row_encoding_version"] = FEREBUS_ROW_ENCODING_VERSION
    updated["contract_sha256"] = canonical_json_sha256(updated)
    atomic_write_json(path, updated)
    return read_feature_contract(path)


def ensure_feature_contract(campaign_dir: Path, config: Any, pointdir: Path) -> Dict[str, Any]:
    campaign = Path(campaign_dir).resolve()
    expected = _build_contract(campaign, config, Path(pointdir))
    path = feature_contract_path(campaign)
    if path.exists():
        observed = read_feature_contract(path)
        if int(observed["row_encoding_version"]) < FEREBUS_ROW_ENCODING_VERSION:
            _preserve_feature_contract(campaign, observed)
            atomic_write_json(path, expected)
            return read_feature_contract(path)
        if observed != expected:
            raise ValueError(
                "staged geometry/config does not match the immutable FEREBUS feature contract"
            )
        return observed
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, expected)
    return read_feature_contract(path)


def _task_row_shard_dir(campaign: Path, task: Mapping[str, Any]) -> Path:
    binding = task.get("ferebus_row_shard")
    if not isinstance(binding, dict):
        raise ValueError("AIMAll task lacks its FEREBUS row-shard binding")
    relative = Path(str(binding.get("directory") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("FEREBUS row-shard directory escapes the campaign")
    return campaign_owned_path(campaign, campaign / relative)


def _source_records(pointdir: Path) -> Tuple[str, list[Dict[str, Any]]]:
    geometry = pointdir / "input.gjf"
    if geometry.is_symlink() or not geometry.is_file():
        raise FileNotFoundError("FEREBUS row shard lacks input.gjf")
    int_files = sorted(
        [path for path in pointdir.rglob("*.int") if path.is_file() and not path.is_symlink()],
        key=lambda path: path.relative_to(pointdir).as_posix(),
    )
    if not int_files:
        raise FileNotFoundError("FEREBUS row shard lacks AIMAll INT files")
    return sha256_file(geometry), [
        {
            "path": path.relative_to(pointdir).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in int_files
    ]


def _validate_source_binding_metadata(
    payload: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
) -> None:
    by_path = {
        str(binding.get("path") or ""): binding
        for binding in source_bindings
        if isinstance(binding, Mapping)
    }
    task = by_path.get("AIMALL_TASK.json")
    geometry = by_path.get("input.gjf")
    if not isinstance(task, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("accepted pointdir metadata lacks row-shard source files")
    if str(task.get("sha256") or "") != str(payload.get("aimall_task_sha256") or ""):
        raise ValueError("FEREBUS row-shard AIMAll-task hash mismatch")
    if str(geometry.get("sha256") or "") != str(payload.get("geometry_sha256") or ""):
        raise ValueError("FEREBUS row-shard geometry hash mismatch")
    expected_ints = [
        {
            "path": path,
            "size": int(binding.get("size", -1)),
            "sha256": str(binding.get("sha256") or ""),
        }
        for path, binding in sorted(by_path.items())
        if path.lower().endswith(".int")
    ]
    if not expected_ints or expected_ints != payload.get("int_files"):
        raise ValueError("FEREBUS row-shard INT inventory mismatch")


def write_row_shard(
    campaign_dir: Path,
    pointdir: Path,
    output_dir: Path,
    *,
    contract: Optional[Mapping[str, Any]] = None,
    logical_pointdir_name: Optional[str] = None,
) -> Path:
    campaign = Path(campaign_dir).resolve()
    point_root = campaign_owned_path(campaign, pointdir)
    selected_contract = dict(contract or read_feature_contract(campaign))
    if int(selected_contract.get("row_encoding_version", -1)) != int(
        FEREBUS_ROW_ENCODING_VERSION
    ):
        raise ValueError("FEREBUS row shard requires the current row encoding")
    system_alf = _contract_system_alf(selected_contract)
    extracted = PointDirectory(point_root).feature_property_rows(
        system_alf,
        list(selected_contract["properties"]),
    )
    if list(extracted["atom_names"]) != list(selected_contract["atom_names"]):
        raise ValueError("FEREBUS row shard atom order differs from its contract")
    if (
        list(extracted["feature_headers"]) + list(extracted["property_headers"])
        != list(selected_contract["row_headers"])
    ):
        raise ValueError("FEREBUS row shard headers differ from its contract")
    root = campaign_owned_path(campaign, output_dir)
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise ValueError("FEREBUS row-shard target is not a regular directory")
        _remove_derived_tree(root)
    root.mkdir(parents=True, exist_ok=False)
    atom_names = [str(atom) for atom in selected_contract["atom_names"]]
    matrix = np.vstack(
        [np.asarray(extracted["rows"][atom], dtype=np.float64) for atom in atom_names]
    )
    path = root / FEREBUS_ROW_SHARD_ARRAY
    _atomic_write_npy(path, matrix)
    row_array = {
        "path": path.name,
        "shape": [int(value) for value in matrix.shape],
        "dtype": "float64",
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }
    provenance = read_provenance(point_root)
    allocation = provenance.get("point_allocation")
    if not isinstance(allocation, dict):
        raise ValueError("FEREBUS row shard lacks point-allocation provenance")
    geometry_sha256, int_records = _source_records(point_root)
    task_path = point_root / "AIMALL_TASK.json"
    payload = {
        "schema_version": FEREBUS_ROW_SHARD_SCHEMA_VERSION,
        "row_encoding_version": FEREBUS_ROW_ENCODING_VERSION,
        "campaign_uid": str(provenance["campaign_uid"]),
        "pointdir": str(logical_pointdir_name or point_root.name),
        "candidate_id": str(allocation.get("candidate_id") or ""),
        "feature_contract_sha256": str(selected_contract["contract_sha256"]),
        "atom_names": atom_names,
        "aimall_task_sha256": sha256_file(task_path),
        "geometry_sha256": geometry_sha256,
        "int_files": int_records,
        "row_array": row_array,
        "created_at_iso": _now_iso(),
    }
    manifest = root / FEREBUS_ROW_SHARD
    atomic_write_json(manifest, payload)
    return manifest


def read_row_shard(
    campaign_dir: Path,
    shard_dir: Path,
    *,
    expected_contract_sha256: str,
    expected_candidate_id: Optional[str] = None,
    source_bindings: Optional[Sequence[Mapping[str, Any]]] = None,
    expected_source_pointdir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    campaign = Path(campaign_dir).resolve()
    root = campaign_owned_path(campaign, shard_dir)
    manifest = root / FEREBUS_ROW_SHARD
    payload = load_path(manifest)
    if not isinstance(payload, dict):
        raise ValueError("FEREBUS row-shard receipt must be a JSON object")
    if payload.get("schema_version") != FEREBUS_ROW_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS row-shard schema")
    if payload.get("row_encoding_version") != FEREBUS_ROW_ENCODING_VERSION:
        raise ValueError("FEREBUS row-shard encoding is incompatible")
    if payload.get("feature_contract_sha256") != str(expected_contract_sha256):
        raise ValueError("FEREBUS row-shard feature contract mismatch")
    if expected_candidate_id is not None and payload.get("candidate_id") != str(
        expected_candidate_id
    ):
        raise ValueError("FEREBUS row-shard candidate mismatch")
    if expected_source_pointdir is not None and payload.get("pointdir") != str(
        expected_source_pointdir
    ):
        raise ValueError("FEREBUS row-shard source-point identity mismatch")
    if source_bindings is not None:
        _validate_source_binding_metadata(payload, source_bindings)
    atom_names = payload.get("atom_names")
    if (
        not isinstance(atom_names, list)
        or not atom_names
        or any(not isinstance(atom, str) or not atom for atom in atom_names)
        or len(set(atom_names)) != len(atom_names)
    ):
        raise ValueError("FEREBUS row-shard atom names are invalid")
    record = payload.get("row_array")
    if not isinstance(record, dict):
        raise ValueError("FEREBUS row-shard array record is invalid")
    relative = Path(str(record.get("path") or ""))
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name != FEREBUS_ROW_SHARD_ARRAY
    ):
        raise ValueError("FEREBUS row-shard array path is invalid")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("FEREBUS row-shard array is missing: " + str(path))
    data = path.read_bytes()
    if len(data) != int(record.get("size", -1)):
        raise ValueError("FEREBUS row-shard array size mismatch")
    if hashlib.sha256(data).hexdigest() != str(record.get("sha256") or ""):
        raise ValueError("FEREBUS row-shard array hash mismatch")
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if (
        array.dtype != np.float64
        or array.ndim != 2
        or array.shape[0] != len(atom_names)
        or list(array.shape) != list(record.get("shape") or [])
        or not np.isfinite(array).all()
    ):
        raise ValueError("FEREBUS row-shard array contents are invalid")
    arrays = {
        str(atom): np.asarray(array[index : index + 1], dtype=np.float64)
        for index, atom in enumerate(atom_names)
    }
    return payload, arrays


def row_shard_binding(
    campaign_dir: Path,
    pointdir: Path,
    *,
    source_bindings: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    campaign = Path(campaign_dir).resolve()
    try:
        task = load_path(Path(pointdir) / "AIMALL_TASK.json")
        if not isinstance(task, dict):
            return None
        root = _task_row_shard_dir(campaign, task)
        contract = read_feature_contract(campaign)
        provenance = read_provenance(pointdir)
        allocation = provenance.get("point_allocation") or {}
        read_row_shard(
            campaign,
            root,
            expected_contract_sha256=str(contract["contract_sha256"]),
            expected_candidate_id=str(allocation.get("candidate_id") or ""),
            source_bindings=source_bindings,
            expected_source_pointdir=Path(pointdir).name,
        )
        manifest = root / FEREBUS_ROW_SHARD
        return {
            "path": root.relative_to(campaign).as_posix(),
            "manifest_sha256": sha256_file(manifest),
            "feature_contract_sha256": str(contract["contract_sha256"]),
        }
    except Exception:
        return None


def produce_task_row_shard(campaign_dir: Path, pointdir: Path) -> Path:
    campaign = Path(campaign_dir).resolve()
    point_root = campaign_owned_path(campaign, pointdir)
    task = load_path(point_root / "AIMALL_TASK.json")
    if not isinstance(task, dict):
        raise ValueError("AIMAll task metadata must be a JSON object")
    root = _task_row_shard_dir(campaign, task)
    contract = read_feature_contract(campaign)
    binding = task.get("ferebus_feature_contract")
    if not isinstance(binding, dict) or binding.get("contract_sha256") != str(
        contract["contract_sha256"]
    ):
        raise ValueError("AIMAll task feature-contract binding mismatch")
    return write_row_shard(campaign, point_root, root, contract=contract)


def _cache_manifest(path: Path) -> Dict[str, Any]:
    payload = load_path(path / FEREBUS_ROW_CACHE)
    if not isinstance(payload, dict):
        raise ValueError("FEREBUS row-cache manifest must be a JSON object")
    if payload.get("schema_version") != FEREBUS_ROW_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported FEREBUS row-cache schema")
    return payload


def read_version_row_cache(
    campaign_dir: Path,
    contract_sha256: str,
    version: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    campaign = Path(campaign_dir).resolve()
    root = campaign_owned_path(
        campaign,
        row_cache_path(campaign, contract_sha256, version),
    )
    contract = read_feature_contract(campaign_dir)
    if contract.get("row_encoding_version") != FEREBUS_ROW_ENCODING_VERSION:
        raise ValueError("FEREBUS row-cache contract encoding is incompatible")
    if str(contract["contract_sha256"]) != str(contract_sha256):
        raise ValueError("FEREBUS row-cache request does not match the active contract")
    payload = _cache_manifest(root)
    if payload.get("row_encoding_version") != FEREBUS_ROW_ENCODING_VERSION:
        raise ValueError("FEREBUS row-cache encoding is incompatible")
    if payload.get("feature_contract_sha256") != str(contract_sha256):
        raise ValueError("FEREBUS row-cache feature contract mismatch")
    if payload.get("reference_data_version") != int(version):
        raise ValueError("FEREBUS row-cache version mismatch")
    arrays: Dict[str, np.ndarray] = {}
    records = payload.get("arrays")
    if not isinstance(records, dict) or set(records) != set(contract["atom_names"]):
        raise ValueError("FEREBUS row-cache arrays are missing")
    identities = payload.get("row_identities")
    if (
        not isinstance(identities, list)
        or len(identities) != int(payload.get("n_rows", -1))
        or payload.get("row_identities_sha256")
        != canonical_json_sha256(identities)
    ):
        raise ValueError("FEREBUS row-cache row identities are invalid")
    identity_keys = []
    for identity in identities:
        if not isinstance(identity, dict):
            raise ValueError("FEREBUS row-cache row identity is not an object")
        key = (
            str(identity.get("pointdir_name") or ""),
            str(identity.get("candidate_id") or ""),
        )
        if not all(key):
            raise ValueError("FEREBUS row-cache row identity is incomplete")
        identity_keys.append(key)
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("FEREBUS row-cache row identities are duplicated")
    for atom, record in records.items():
        relative = Path(str(record.get("path") or ""))
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
        ):
            raise ValueError("FEREBUS row-cache array path is invalid")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("FEREBUS row-cache array is missing")
        if int(path.stat().st_size) != int(record.get("size", -1)):
            raise ValueError("FEREBUS row-cache array size mismatch")
        if sha256_file(path) != str(record.get("sha256") or ""):
            raise ValueError("FEREBUS row-cache array hash mismatch")
        array = np.load(path, allow_pickle=False)
        if (
            array.dtype != np.float64
            or array.ndim != 2
            or list(array.shape) != list(record.get("shape") or [])
            or not np.isfinite(array).all()
        ):
            raise ValueError("FEREBUS row-cache array contents are invalid")
        arrays[str(atom)] = array
    if any(array.shape[0] != int(payload.get("n_rows", -1)) for array in arrays.values()):
        raise ValueError("FEREBUS row-cache row count mismatch")
    return payload, arrays


def build_version_row_cache(
    campaign_dir: Path,
    *,
    version: int,
    point_bindings: Sequence[Mapping[str, Any]],
    repair_root: Path,
    progress_callback: RowProgressCallback = None,
    reuse_acceptance_shards: bool = True,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir).resolve()
    contract = ensure_current_feature_contract(campaign)
    contract_sha = str(contract["contract_sha256"])
    target = campaign_owned_path(
        campaign,
        row_cache_path(campaign, contract_sha, int(version)),
    )
    if target.is_symlink():
        raise ValueError("FEREBUS row-cache target is symlinked")
    if target.exists():
        if target.is_dir():
            try:
                manifest, _ = read_version_row_cache(
                    campaign, contract_sha, int(version)
                )
                return {**manifest, "created": False}
            except Exception:
                pass
        quarantine = target.with_name(
            target.name + ".invalid." + uuid.uuid4().hex[:12]
        )
        os.replace(str(target), str(quarantine))
        _fsync_parent_dir(quarantine)
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError("FEREBUS row-cache staging is invalid")
        _remove_derived_tree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    rows = {str(atom): [] for atom in contract["atom_names"]}
    identities = []
    repaired = 0
    reused = 0
    repair_base = Path(repair_root)
    repair_base.mkdir(parents=True, exist_ok=True)
    for row_index, binding in enumerate(point_bindings):
        pointdir = Path(str(binding["pointdir_path"]))
        candidate_id = str(binding["candidate_id"])
        receipt = load_path(pointdir / "QUANTUM_ACCEPTANCE_RECEIPT.json")
        shard_binding = receipt.get("ferebus_row_shard") if isinstance(receipt, dict) else None
        source_bindings = receipt.get("artefacts") if isinstance(receipt, dict) else None
        shard_root = None
        shard_payload = None
        shard_rows = None
        if reuse_acceptance_shards and isinstance(shard_binding, dict):
            candidate = campaign / str(shard_binding.get("path") or "")
            try:
                shard_payload, shard_rows = read_row_shard(
                    campaign,
                    candidate,
                    expected_contract_sha256=contract_sha,
                    expected_candidate_id=candidate_id,
                    source_bindings=source_bindings,
                    expected_source_pointdir=str(binding["source_pointdir"]),
                )
                if sha256_file(candidate / FEREBUS_ROW_SHARD) != str(
                    shard_binding.get("manifest_sha256") or ""
                ):
                    raise ValueError("FEREBUS row-shard receipt hash mismatch")
                shard_root = candidate
                reused += 1
            except Exception:
                shard_root = None
        if shard_root is None:
            # Keep repair names short so crash-recovery fixtures remain portable to
            # Windows while the manifest retains the canonical point identity.
            shard_root = repair_base / ("row-" + str(row_index).zfill(6))
            try:
                shard_payload, shard_rows = read_row_shard(
                    campaign,
                    shard_root,
                    expected_contract_sha256=contract_sha,
                    expected_candidate_id=candidate_id,
                    source_bindings=source_bindings,
                    expected_source_pointdir=str(binding["source_pointdir"]),
                )
            except Exception:
                write_row_shard(
                    campaign,
                    pointdir,
                    shard_root,
                    contract=contract,
                    logical_pointdir_name=str(binding["source_pointdir"]),
                )
                repaired += 1
                shard_payload, shard_rows = read_row_shard(
                    campaign,
                    shard_root,
                    expected_contract_sha256=contract_sha,
                    expected_candidate_id=candidate_id,
                    source_bindings=source_bindings,
                    expected_source_pointdir=str(binding["source_pointdir"]),
                )
        if shard_payload is None or shard_rows is None:
            raise ValueError("FEREBUS row-shard validation produced no rows")
        if list(shard_payload.get("atom_names") or []) != list(
            contract["atom_names"]
        ):
            raise ValueError("FEREBUS row-shard atom order differs from its contract")
        if set(shard_rows) != set(rows):
            raise ValueError("FEREBUS row-shard atom set differs from its contract")
        for atom in contract["atom_names"]:
            rows[str(atom)].append(shard_rows[str(atom)][0])
        identities.append(
            {
                "pointdir_name": str(binding["pointdir_name"]),
                "source_pointdir": str(binding["source_pointdir"]),
                "candidate_id": candidate_id,
                "row_shard_sha256": sha256_file(shard_root / FEREBUS_ROW_SHARD),
                "row_shard_created_at_iso": str(shard_payload.get("created_at_iso") or ""),
            }
        )
        processed = row_index + 1
        if progress_callback is not None and (
            processed % ROW_CACHE_PROGRESS_INTERVAL == 0
            or processed == len(point_bindings)
        ):
            progress_callback(
                "reference_commit_shard_progress",
                {
                    "processed_points": int(processed),
                    "total_points": int(len(point_bindings)),
                    "shards_reused": int(reused),
                    "shards_repaired": int(repaired),
                },
            )
    arrays: Dict[str, Dict[str, Any]] = {}
    for atom in contract["atom_names"]:
        matrix = np.asarray(rows[str(atom)], dtype=np.float64)
        path = staging / (str(atom) + ".npy")
        _atomic_write_npy(path, matrix)
        arrays[str(atom)] = {
            "path": path.name,
            "shape": [int(value) for value in matrix.shape],
            "dtype": "float64",
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    payload = {
        "schema_version": FEREBUS_ROW_CACHE_SCHEMA_VERSION,
        "row_encoding_version": FEREBUS_ROW_ENCODING_VERSION,
        "feature_contract_sha256": contract_sha,
        "reference_data_version": int(version),
        "n_rows": len(identities),
        "row_identities": identities,
        "row_identities_sha256": canonical_json_sha256(identities),
        "arrays": arrays,
        "shards_reused": int(reused),
        "shards_repaired": int(repaired),
        "completed_at_iso": _now_iso(),
    }
    atomic_write_json(staging / FEREBUS_ROW_CACHE, payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(staging), str(target))
    _fsync_parent_dir(target)
    validated, _ = read_version_row_cache(campaign, contract_sha, int(version))
    return {**validated, "created": True}


def load_cumulative_rows(
    campaign_dir: Path,
    version: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], list[Dict[str, Any]]]:
    campaign = Path(campaign_dir)
    contract = read_feature_contract(campaign)
    contract_sha = str(contract["contract_sha256"])
    rows = {str(atom): [] for atom in contract["atom_names"]}
    identities: list[Dict[str, Any]] = []
    for current in range(int(version) + 1):
        manifest, arrays = read_version_row_cache(campaign, contract_sha, current)
        for atom in contract["atom_names"]:
            rows[str(atom)].append(arrays[str(atom)])
        identities.extend(list(manifest["row_identities"]))
    combined = {
        atom: np.concatenate(chunks, axis=0)
        for atom, chunks in rows.items()
    }
    if any(array.shape[0] != len(identities) for array in combined.values()):
        raise ValueError("cumulative FEREBUS row-cache identity count mismatch")
    return contract, combined, identities


def ensure_cumulative_row_caches(
    campaign_dir: Path,
    view: Any,
) -> None:
    """Rebuild derived caches serially from authoritative committed pointdirs."""
    campaign = Path(campaign_dir)
    contract = ensure_current_feature_contract(campaign)
    contract_sha = str(contract["contract_sha256"])
    for version in range(int(view.version) + 1):
        try:
            read_version_row_cache(campaign, contract_sha, version)
            continue
        except Exception:
            target = row_cache_path(campaign, contract_sha, version)
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_dir():
                    raise ValueError(
                        "invalid FEREBUS row-cache target cannot be rebuilt safely: "
                        + str(target)
                    )
                quarantine = target.with_name(
                    target.name + ".invalid." + uuid.uuid4().hex[:12]
                )
                os.replace(str(target), str(quarantine))
                _fsync_parent_dir(quarantine)
        entries = [
            entry for entry in view.entries if int(entry.introduced_in_version) == version
        ]
        bindings = [
            {
                "pointdir_path": entry.pointdir_path,
                "pointdir_name": entry.pointdir_name,
                "source_pointdir": entry.source_pointdir,
                "candidate_id": entry.candidate_id,
            }
            for entry in entries
        ]
        repair_root = (
            campaign
            / ".DATA"
            / "CACHE"
            / "FEREBUS_ROWS"
            / "_repair"
            / (contract_sha[:12] + "-v" + str(version).zfill(6))
        )
        repair_root.mkdir(parents=True, exist_ok=True)
        build_version_row_cache(
            campaign,
            version=version,
            point_bindings=bindings,
            repair_root=repair_root,
            reuse_acceptance_shards=False,
        )


def clear_row_caches(campaign_dir: Path) -> bool:
    """Remove only derived FEREBUS row caches during explicit repair."""
    campaign = Path(campaign_dir).resolve()
    root = campaign / ".DATA" / "CACHE" / "FEREBUS_ROWS"
    if not root.exists() and not root.is_symlink():
        return False
    if root.is_symlink():
        raise ValueError("FEREBUS row-cache root is symlinked")
    owned = campaign_owned_path(campaign, root)
    if not owned.is_dir():
        raise ValueError("FEREBUS row-cache root is not a regular directory")
    _remove_derived_tree(owned)
    _fsync_parent_dir(owned)
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("produce",))
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--pointdir", required=True)
    args = parser.parse_args(argv)
    produce_task_row_shard(Path(args.campaign_dir), Path(args.pointdir))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by generated scripts
    raise SystemExit(main())


__all__ = [
    "FEREBUS_FEATURE_CONTRACT",
    "FEREBUS_FEATURE_CONTRACT_SCHEMA_VERSION",
    "FEREBUS_ROW_CACHE",
    "FEREBUS_ROW_CACHE_SCHEMA_VERSION",
    "FEREBUS_ROW_ENCODING_VERSION",
    "FEREBUS_ROW_SHARD",
    "FEREBUS_ROW_SHARD_ARRAY",
    "FEREBUS_ROW_SHARD_SCHEMA_VERSION",
    "build_version_row_cache",
    "clear_row_caches",
    "ensure_feature_contract",
    "ensure_current_feature_contract",
    "ensure_cumulative_row_caches",
    "feature_contract_path",
    "load_cumulative_rows",
    "produce_task_row_shard",
    "read_feature_contract",
    "read_row_shard",
    "read_version_row_cache",
    "row_cache_path",
    "row_shard_binding",
    "system_alf_from_contract",
    "write_row_shard",
]
