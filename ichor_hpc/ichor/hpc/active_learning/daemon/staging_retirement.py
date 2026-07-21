"""Retire staging buckets that are fully represented by committed authority."""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..layout import (
    parse_staging_pointdir_name,
    qm_reference_data_dir,
    staging_root,
)
from ..strict_json import load_path
from ..versioning.manifest import sha256_file
from ..versioning.reference_data import ReferenceDataVersioning
from .ferebus_row_cache import (
    FEREBUS_ROW_SHARD,
    FEREBUS_ROW_SHARD_ARRAY,
    read_feature_contract,
    read_row_shard,
    read_version_row_cache,
)
from .quantum_quality import read_quantum_quality_manifest
from .state import _fsync_parent_dir


STAGING_RETIRED_DIRNAME = "STAGING_RETIRED"

_ACTIVE_BUCKET_RE = re.compile(r"^iter_([0-9]+)$")
_REPLACEMENT_ROUND_RE = re.compile(r"^replacement_round_([0-9]{4,})$")
_ACCEPTANCE_MANIFEST_RE = re.compile(
    r"^accepted_pointdirs(?:\.([A-Z][A-Z0-9_]*))?\.json$"
)
_LOWER_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StagingRetirementError(ValueError):
    """Raised when completed staging cannot be retired without ambiguity."""


def retired_staging_root(campaign_dir: Path) -> Path:
    return Path(campaign_dir) / ".DATA" / STAGING_RETIRED_DIRNAME


def _bucket_identity(path: Path) -> Optional[Tuple[str, int]]:
    if path.name == "initial":
        return "bootstrap", 0
    match = _ACTIVE_BUCKET_RE.fullmatch(path.name)
    if match is None:
        return None
    iteration = int(match.group(1))
    if iteration < 1 or path.name != "iter_" + str(iteration):
        return None
    return "active", iteration


def _required_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise StagingRetirementError(label + " is not a regular file: " + str(path))


def _validate_authority(
    campaign: Path,
    bucket: Path,
    *,
    context: str,
    iteration: int,
) -> Dict[str, Any]:
    # Local import avoids a module cycle: reference publication invokes this
    # service after its own ledger has become complete.
    from .reference_commit import classify_reference_commit

    transaction = classify_reference_commit(
        campaign,
        context=context,
        iteration=int(iteration),
        verification="authority",
    )
    if str(transaction.get("state") or "") != "complete":
        raise StagingRetirementError(
            "staging bucket lacks a complete reference-commit transaction"
        )
    ledger = transaction.get("ledger")
    if not isinstance(ledger, Mapping):
        raise StagingRetirementError("completed reference transaction lacks its ledger")
    transaction_id = str(ledger.get("transaction_id") or "")
    if _LOWER_SHA256_RE.fullmatch(transaction_id) is None:
        raise StagingRetirementError("reference transaction ID is invalid")
    bindings = ledger.get("point_bindings")
    if (
        not isinstance(bindings, list)
        or not bindings
        or int(ledger.get("moved_points", -1)) != len(bindings)
        or any(
            not isinstance(record, Mapping)
            or str(record.get("move_state") or "") != "destination"
            for record in bindings
        )
    ):
        raise StagingRetirementError(
            "completed reference transaction does not bind every moved point"
        )

    versioning = ReferenceDataVersioning(qm_reference_data_dir(campaign))
    current = versioning.current_version()
    if current is None or int(current) < int(iteration):
        raise StagingRetirementError(
            "reference-data current pointer has not advanced through this bucket"
        )
    view = versioning.resolve(int(iteration), verification="index")
    committed = {
        entry.pointdir_name: entry
        for entry in view.entries
        if int(entry.introduced_in_version) == int(iteration)
    }
    if len(committed) != len(bindings):
        raise StagingRetirementError(
            "committed reference-data delta does not match the transaction"
        )

    final = versioning.iteration_path(int(iteration))
    committed_receipts: Dict[str, Mapping[str, Any]] = {}
    source_bindings: Dict[str, Mapping[str, Any]] = {}
    for record in bindings:
        destination_name = str(record.get("destination_pointdir") or "")
        entry = committed.get(destination_name)
        if entry is None:
            raise StagingRetirementError(
                "reference transaction point is absent from committed metadata"
            )
        expected = {
            "pointdir_name": destination_name,
            "source_pointdir": str(record.get("source_pointdir") or ""),
            "candidate_id": str(record.get("candidate_id") or ""),
            "slot_id": int(record.get("slot_id", -1)),
            "split": str(record.get("split") or ""),
            "replacement_round": int(record.get("replacement_round", -1)),
            "accepted_content_sha256": str(
                record.get("accepted_content_sha256") or ""
            ),
            "acceptance_receipt_sha256": str(
                record.get("acceptance_receipt_sha256") or ""
            ),
            "provenance_sha256": str(record.get("provenance_sha256") or ""),
        }
        observed = {
            key: entry.identity_payload()[key]
            for key in expected
        }
        if observed != expected:
            raise StagingRetirementError(
                "committed reference-data identity differs from its transaction"
            )
        pointdir = final / destination_name
        if pointdir.is_symlink() or not pointdir.is_dir():
            raise StagingRetirementError(
                "committed reference pointdir is missing: " + str(pointdir)
            )
        receipt_path = pointdir / "QUANTUM_ACCEPTANCE_RECEIPT.json"
        _required_regular_file(receipt_path, "committed acceptance receipt")
        receipt = load_path(receipt_path)
        if not isinstance(receipt, Mapping):
            raise StagingRetirementError(
                "committed acceptance receipt is not a JSON object"
            )
        if sha256_file(receipt_path) != str(entry.acceptance_receipt_sha256):
            raise StagingRetirementError(
                "committed acceptance receipt differs from reference authority"
            )
        if (
            str(receipt.get("source_pointdir") or "")
            != str(record.get("source_pointdir") or "")
            or str(receipt.get("candidate_id") or "")
            != str(record.get("candidate_id") or "")
            or int(receipt.get("iteration", -1)) != int(iteration)
            or str(receipt.get("content_sha256") or "")
            != str(record.get("accepted_content_sha256") or "")
        ):
            raise StagingRetirementError(
                "committed acceptance receipt identity differs from its transaction"
            )
        quality_binding = receipt.get("quality_manifest")
        if (
            not isinstance(quality_binding, Mapping)
            or str(quality_binding.get("path") or "")
            != str(record.get("quality_manifest_path") or "")
            or str(quality_binding.get("sha256") or "")
            != str(record.get("quality_manifest_sha256") or "")
        ):
            raise StagingRetirementError(
                "committed acceptance receipt quality binding is invalid"
            )
        artefacts = receipt.get("artefacts")
        if not isinstance(artefacts, list) or not artefacts:
            raise StagingRetirementError(
                "committed acceptance receipt lacks its artefact inventory"
            )
        source_name = str(record.get("source_pointdir") or "")
        if not source_name or source_name in source_bindings:
            raise StagingRetirementError(
                "reference transaction source point identities are invalid"
            )
        committed_receipts[source_name] = receipt
        source_bindings[source_name] = record

    contract = read_feature_contract(campaign)
    contract_sha = str(contract.get("contract_sha256") or "")
    row_cache_state = "valid"
    row_cache_error = None
    try:
        read_version_row_cache(campaign, contract_sha, int(iteration))
    except Exception as exc:
        # A complete committed pointdir inventory is the deterministic repair
        # source used by ensure_cumulative_row_caches(). The derived cache need
        # not block retirement merely because it must be rebuilt later.
        row_cache_state = "rebuildable"
        row_cache_error = type(exc).__name__ + ": " + str(exc)

    return {
        "bucket": str(bucket),
        "context": str(context),
        "iteration": int(iteration),
        "reference_data_version": int(iteration),
        "transaction_id": transaction_id,
        "ledger_path": str(transaction.get("path") or ""),
        "ledger": dict(ledger),
        "bindings_by_source": source_bindings,
        "receipts_by_source": committed_receipts,
        "feature_contract_sha256": contract_sha,
        "row_cache_state": row_cache_state,
        "row_cache_error": row_cache_error,
    }


def _pointdir_name(value: str) -> str:
    parse_staging_pointdir_name(str(value))
    return str(value)


def _validate_points_file(
    path: Path,
    *,
    container: Path,
    committed_sources: set[str],
) -> List[str]:
    _required_regular_file(path, "staging POINTS.txt")
    reasons: List[str] = []
    seen = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw.strip()
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            raise StagingRetirementError(
                "POINTS.txt contains a relative path at line " + str(line_number)
            )
        name = _pointdir_name(candidate.name)
        expected = (container / name).resolve(strict=False)
        if candidate.resolve(strict=False) != expected or name in seen:
            raise StagingRetirementError(
                "POINTS.txt contains a non-canonical or duplicate path"
            )
        seen.add(name)
        if name not in committed_sources:
            reasons.append("POINTS.txt references uncommitted pointdir " + name)
    return reasons


def _validate_acceptance_manifest(
    path: Path,
    *,
    iteration: int,
    committed_sources: set[str],
) -> List[str]:
    _required_regular_file(path, "quantum acceptance manifest")
    payload = load_path(path)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 2:
        raise StagingRetirementError("quantum acceptance manifest is malformed")
    if int(payload.get("iteration", -1)) != int(iteration):
        raise StagingRetirementError("quantum acceptance manifest iteration mismatch")
    phase = str(payload.get("phase") or "")
    match = _ACCEPTANCE_MANIFEST_RE.fullmatch(path.name)
    if match is None or (match.group(1) is not None and match.group(1) != phase):
        raise StagingRetirementError("quantum acceptance manifest phase mismatch")
    accepted = payload.get("accepted_pointdirs")
    rejected = payload.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise StagingRetirementError("quantum acceptance disposition is malformed")
    accepted_names = [_pointdir_name(str(name)) for name in accepted]
    rejected_names = []
    for record in rejected:
        if not isinstance(record, Mapping) or not str(record.get("reason") or ""):
            raise StagingRetirementError("quantum rejection record is malformed")
        rejected_names.append(_pointdir_name(str(record.get("pointdir") or "")))
    if (
        len(accepted_names) != len(set(accepted_names))
        or len(rejected_names) != len(set(rejected_names))
        or set(accepted_names) & set(rejected_names)
        or int(payload.get("n_total", -1))
        != len(accepted_names) + len(rejected_names)
    ):
        raise StagingRetirementError("quantum acceptance counts are inconsistent")
    reasons = [
        "acceptance manifest records rejected pointdir " + name
        for name in rejected_names
    ]
    reasons.extend(
        "acceptance manifest references uncommitted pointdir " + name
        for name in accepted_names
        if name not in committed_sources
    )
    return reasons


def _validate_quality_manifest(
    campaign: Path,
    path: Path,
    *,
    iteration: int,
    committed_sources: set[str],
    authority: Mapping[str, Any],
) -> List[str]:
    _required_regular_file(path, "quantum-quality manifest")
    try:
        relative = path.resolve(strict=False).relative_to(
            campaign.resolve()
        ).as_posix()
    except ValueError as exc:
        raise StagingRetirementError(
            "quantum-quality manifest escapes the campaign"
        ) from exc
    bound_records = [
        record
        for record in authority["ledger"]["point_bindings"]
        if str(record.get("quality_manifest_path") or "") == relative
    ]
    if not bound_records:
        return ["unbound quantum-quality manifest: " + path.name]
    expected_hashes = {
        str(record.get("quality_manifest_sha256") or "")
        for record in bound_records
    }
    if len(expected_hashes) != 1 or sha256_file(path) not in expected_hashes:
        raise StagingRetirementError(
            "quantum-quality manifest differs from committed authority"
        )
    raw = load_path(path)
    if not isinstance(raw, Mapping):
        raise StagingRetirementError("quantum-quality manifest is malformed")
    quality = read_quantum_quality_manifest(
        path.parent,
        expected_phase=str(raw.get("phase") or ""),
        expected_iteration=int(iteration),
        manifest_path=path,
    )
    reasons = []
    for record in quality["records"]:
        name = _pointdir_name(str(record.get("pointdir") or ""))
        if not bool(record.get("accepted", False)):
            reasons.append("quantum quality rejected pointdir " + name)
        elif name not in committed_sources:
            reasons.append("quantum quality references uncommitted pointdir " + name)
    return reasons


def _validate_replacement_round(
    campaign: Path,
    round_dir: Path,
    *,
    context: str,
    iteration: int,
    authority: Mapping[str, Any],
) -> List[str]:
    from ..replacement_sampling import read_replacement_sample

    match = _REPLACEMENT_ROUND_RE.fullmatch(round_dir.name)
    if match is None:
        raise StagingRetirementError("replacement round name is non-canonical")
    replacement_round = int(match.group(1))
    if round_dir.name != "replacement_round_" + str(replacement_round).zfill(4):
        raise StagingRetirementError("replacement round name is non-canonical")
    sample = read_replacement_sample(round_dir, verify_allocation=False)
    if (
        str(sample.get("context") or "") != str(context)
        or int(sample.get("iteration", -1)) != int(iteration)
        or int(sample.get("replacement_round", -1)) != replacement_round
    ):
        raise StagingRetirementError("replacement sample identity mismatch")
    expected_allocation = Path(
        str(authority["ledger"].get("point_allocation_path") or "")
    )
    if not expected_allocation.is_absolute():
        expected_allocation = campaign / expected_allocation
    if Path(str(sample["point_allocation_manifest"])).resolve(strict=False) != (
        expected_allocation.resolve(strict=False)
    ):
        raise StagingRetirementError("replacement sample allocation path mismatch")
    bindings = authority["bindings_by_source"]
    reasons = []
    for record in sample["records"]:
        candidate_id = str(record.get("candidate_id") or "")
        matches = [
            binding
            for binding in bindings.values()
            if str(binding.get("candidate_id") or "") == candidate_id
            and int(binding.get("slot_id", -1)) == int(record.get("slot_id", -2))
            and str(binding.get("split") or "") == str(record.get("split") or "")
            and int(binding.get("replacement_round", -1)) == replacement_round
        ]
        if len(matches) != 1:
            reasons.append(
                "replacement sample contains uncommitted candidate " + candidate_id
            )
    return reasons


def _validate_row_shards(
    campaign: Path,
    root: Path,
    *,
    authority: Mapping[str, Any],
) -> List[str]:
    if root.is_symlink() or not root.is_dir():
        return ["FEREBUS row-shard root is not a regular directory"]
    reasons = []
    expected_contract = str(authority["feature_contract_sha256"])
    bindings = authority["bindings_by_source"]
    receipts = authority["receipts_by_source"]
    expected_paths: Dict[Path, str] = {}
    for source_name, receipt in receipts.items():
        shard_binding = receipt.get("ferebus_row_shard")
        if not isinstance(shard_binding, Mapping):
            continue
        relative = Path(str(shard_binding.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise StagingRetirementError(
                "committed acceptance receipt has an invalid row-shard path"
            )
        candidate = (campaign / relative).resolve(strict=False)
        if candidate.parent == root.resolve(strict=False):
            expected_paths[candidate] = source_name
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            reasons.append("FEREBUS row-shard entry is symlinked: " + child.name)
            continue
        source_name = expected_paths.get(child.resolve(strict=False))
        if source_name is None or not child.is_dir():
            reasons.append("unrecognised FEREBUS row-shard entry: " + child.name)
            continue
        _pointdir_name(source_name)
        binding = bindings.get(source_name)
        receipt = receipts.get(source_name)
        if not isinstance(binding, Mapping) or not isinstance(receipt, Mapping):
            reasons.append("row shard belongs to uncommitted pointdir " + source_name)
            continue
        shard_binding = receipt.get("ferebus_row_shard")
        if (
            not isinstance(shard_binding, Mapping)
            or str(shard_binding.get("feature_contract_sha256") or "")
            != expected_contract
        ):
            raise StagingRetirementError(
                "committed acceptance receipt row-shard binding is invalid"
            )
        observed_names = sorted(item.name for item in child.iterdir())
        if observed_names != sorted([FEREBUS_ROW_SHARD, FEREBUS_ROW_SHARD_ARRAY]):
            reasons.append("row shard contains unrecognised payloads: " + child.name)
            continue
        try:
            read_row_shard(
                campaign,
                child,
                expected_contract_sha256=expected_contract,
                expected_candidate_id=str(binding.get("candidate_id") or ""),
                source_bindings=receipt.get("artefacts"),
                expected_source_pointdir=source_name,
            )
            if sha256_file(child / FEREBUS_ROW_SHARD) != str(
                shard_binding.get("manifest_sha256") or ""
            ):
                raise ValueError("FEREBUS row-shard receipt hash mismatch")
        except Exception as exc:
            raise StagingRetirementError(
                "FEREBUS row shard is invalid: "
                + child.name
                + ": "
                + type(exc).__name__
                + ": "
                + str(exc)
            ) from exc
    return reasons


def _classify_regular_file(
    campaign: Path,
    child: Path,
    *,
    container: Path,
    iteration: int,
    committed_sources: set[str],
    replacement_round: bool,
    authority: Mapping[str, Any],
) -> Tuple[List[str], bool]:
    if child.name == "POINTS.txt":
        return (
            _validate_points_file(
                child,
                container=container,
                committed_sources=committed_sources,
            ),
            False,
        )
    if _ACCEPTANCE_MANIFEST_RE.fullmatch(child.name):
        return (
            _validate_acceptance_manifest(
                child,
                iteration=int(iteration),
                committed_sources=committed_sources,
            ),
            False,
        )
    if child.name == "quantum_quality.json":
        return (
            _validate_quality_manifest(
                campaign,
                child,
                iteration=int(iteration),
                committed_sources=committed_sources,
                authority=authority,
            ),
            False,
        )
    if replacement_round and child.name in {
        "REPLACEMENT_SAMPLE.json",
        "replacement-SAMPLE.xyz",
    }:
        return [], True
    return ["unrecognised file: " + child.name], False


def _classify_container(
    campaign: Path,
    container: Path,
    *,
    context: str,
    iteration: int,
    authority: Mapping[str, Any],
    replacement_round: bool,
) -> List[str]:
    reasons: List[str] = []
    committed_sources = set(authority["bindings_by_source"])
    expected_round_files = {"REPLACEMENT_SAMPLE.json", "replacement-SAMPLE.xyz"}
    observed_round_files = set()
    for child in sorted(container.iterdir(), key=lambda item: item.name):
        try:
            mode = child.lstat().st_mode
        except OSError as exc:
            raise StagingRetirementError(
                "staging entry cannot be inspected: " + str(child)
            ) from exc
        if stat.S_ISLNK(mode):
            reasons.append("nested symlink: " + child.relative_to(container).as_posix())
            continue
        if stat.S_ISDIR(mode):
            if child.name in {"ferebus_row_shards", ".ferebus-row-shards"}:
                reasons.extend(
                    _validate_row_shards(campaign, child, authority=authority)
                )
            elif child.name.endswith(".pointdir"):
                _pointdir_name(child.name)
                reasons.append("retained pointdir payload: " + child.name)
            else:
                reasons.append("unrecognised directory: " + child.name)
            continue
        if not stat.S_ISREG(mode):
            reasons.append("special file: " + child.name)
            continue
        file_reasons, is_round_file = _classify_regular_file(
            campaign,
            child,
            container=container,
            iteration=int(iteration),
            committed_sources=committed_sources,
            replacement_round=replacement_round,
            authority=authority,
        )
        reasons.extend(file_reasons)
        if is_round_file:
            observed_round_files.add(child.name)
    if replacement_round:
        if observed_round_files != expected_round_files:
            raise StagingRetirementError(
                "replacement round metadata is missing or incomplete"
            )
        reasons.extend(
            _validate_replacement_round(
                campaign,
                container,
                context=context,
                iteration=int(iteration),
                authority=authority,
            )
        )
    return reasons


def _classify_bucket_contents(
    campaign: Path,
    bucket: Path,
    *,
    context: str,
    iteration: int,
    authority: Mapping[str, Any],
) -> List[str]:
    reasons: List[str] = []
    committed_sources = set(authority["bindings_by_source"])
    for child in sorted(bucket.iterdir(), key=lambda item: item.name):
        try:
            mode = child.lstat().st_mode
        except OSError as exc:
            raise StagingRetirementError(
                "staging entry cannot be inspected: " + str(child)
            ) from exc
        if stat.S_ISLNK(mode):
            reasons.append("nested symlink: " + child.name)
            continue
        if stat.S_ISDIR(mode) and _REPLACEMENT_ROUND_RE.fullmatch(child.name):
            reasons.extend(
                _classify_container(
                    campaign,
                    child,
                    context=context,
                    iteration=int(iteration),
                    authority=authority,
                    replacement_round=True,
                )
            )
            continue
        if stat.S_ISDIR(mode) and child.name in {
            "ferebus_row_shards",
            ".ferebus-row-shards",
        }:
            reasons.extend(_validate_row_shards(campaign, child, authority=authority))
            continue
        if stat.S_ISDIR(mode) and child.name.endswith(".pointdir"):
            _pointdir_name(child.name)
            reasons.append("retained pointdir payload: " + child.name)
            continue
        if stat.S_ISDIR(mode):
            reasons.append("unrecognised directory: " + child.name)
            continue
        if not stat.S_ISREG(mode):
            reasons.append("special file: " + child.name)
            continue
        file_reasons, _is_round_file = _classify_regular_file(
            campaign,
            child,
            container=bucket,
            iteration=int(iteration),
            committed_sources=committed_sources,
            replacement_round=False,
            authority=authority,
        )
        reasons.extend(file_reasons)
    return sorted(set(reasons))


def classify_completed_staging_buckets(
    campaign_dir: Path,
    *,
    through_version: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify canonical staging buckets without mutating campaign data."""
    campaign = Path(campaign_dir).resolve()
    root = staging_root(campaign)
    result: Dict[str, Any] = {
        "eligible": [],
        "ambiguous": [],
        "noncanonical": [],
        "pending_tombstones": [],
    }
    retired_root = retired_staging_root(campaign)
    if retired_root.is_dir() and not retired_root.is_symlink():
        result["pending_tombstones"] = [
            str(path)
            for path in sorted(
                retired_root.glob(".deleting-*"),
                key=lambda item: item.name,
            )
        ]
    if not root.exists() and not root.is_symlink():
        return result
    if root.is_symlink() or not root.is_dir():
        result["ambiguous"].append(
            {
                "bucket": str(root),
                "state": "ambiguous",
                "reason": "staging root is not a regular directory",
            }
        )
        return result
    for bucket in sorted(root.iterdir(), key=lambda item: item.name):
        identity = _bucket_identity(bucket)
        if identity is None:
            result["noncanonical"].append(str(bucket))
            continue
        context, iteration = identity
        if through_version is not None and int(iteration) > int(through_version):
            continue
        if bucket.is_symlink() or not bucket.is_dir():
            result["ambiguous"].append(
                {
                    "bucket": str(bucket),
                    "context": context,
                    "iteration": int(iteration),
                    "state": "ambiguous",
                    "reason": "canonical staging bucket is not a regular directory",
                }
            )
            continue
        try:
            authority = _validate_authority(
                campaign,
                bucket,
                context=context,
                iteration=int(iteration),
            )
            preserve_reasons = _classify_bucket_contents(
                campaign,
                bucket,
                context=context,
                iteration=int(iteration),
                authority=authority,
            )
        except Exception as exc:
            result["ambiguous"].append(
                {
                    "bucket": str(bucket),
                    "context": context,
                    "iteration": int(iteration),
                    "state": "ambiguous",
                    "reason": type(exc).__name__ + ": " + str(exc),
                }
            )
            continue
        action = "preserve" if preserve_reasons else "delete"
        destination_name = (
            ("bootstrap" if context == "bootstrap" else "active")
            + "-iteration-"
            + str(int(iteration)).zfill(6)
            + "-"
            + str(authority["transaction_id"])
        )
        result["eligible"].append(
            {
                "bucket": str(bucket),
                "context": context,
                "iteration": int(iteration),
                "reference_data_version": int(iteration),
                "transaction_id": str(authority["transaction_id"]),
                "action": action,
                "destination": str(
                    retired_staging_root(campaign)
                    / (
                        ".deleting-" + str(authority["transaction_id"])
                        if action == "delete"
                        else destination_name
                    )
                ),
                "preserve_reasons": preserve_reasons,
                "row_cache_state": str(authority["row_cache_state"]),
            }
        )
    return result


def _remove_tombstone(path: Path) -> Optional[str]:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_dir():
        return "retirement tombstone is not a regular directory: " + str(path)
    try:
        _make_owner_writable_tree(path)
        shutil.rmtree(path)
        _fsync_parent_dir(path)
    except Exception as exc:
        return (
            "could not remove staging retirement tombstone "
            + str(path)
            + ": "
            + type(exc).__name__
            + ": "
            + str(exc)
        )
    return None


def _make_owner_writable_tree(root: Path) -> None:
    """Apply owner ``rwX`` without following retained diagnostic symlinks."""
    pending = [Path(root)]
    while pending:
        path = pending.pop()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            continue
        owner_mode = mode | stat.S_IRUSR | stat.S_IWUSR
        if stat.S_ISDIR(mode):
            owner_mode |= stat.S_IXUSR
        os.chmod(path, stat.S_IMODE(owner_mode))
        if stat.S_ISDIR(mode):
            with os.scandir(path) as entries:
                pending.extend(Path(entry.path) for entry in entries)


def retire_completed_staging_buckets(
    campaign_dir: Path,
    *,
    through_version: Optional[int] = None,
    classification: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically move proven completed buckets out of active staging."""
    campaign = Path(campaign_dir).resolve()
    plan = dict(
        classification
        or classify_completed_staging_buckets(
            campaign,
            through_version=through_version,
        )
    )
    ambiguous = list(plan.get("ambiguous") or [])
    if ambiguous:
        raise StagingRetirementError(
            "completed staging retirement is ambiguous: "
            + "; ".join(str(item.get("reason") or item.get("bucket")) for item in ambiguous)
        )
    eligible = list(plan.get("eligible") or [])
    retired_root = retired_staging_root(campaign)
    result: Dict[str, Any] = {
        "retired": [],
        "deleted": [],
        "preserved": [],
        "warnings": [],
        "n_retired": 0,
        "n_deleted": 0,
        "n_preserved": 0,
    }
    if not eligible and not retired_root.is_dir():
        return result
    if retired_root.is_symlink() or (retired_root.exists() and not retired_root.is_dir()):
        raise StagingRetirementError(
            "staging retirement root is not a regular directory: " + str(retired_root)
        )
    retired_root.mkdir(parents=True, exist_ok=True)
    source_by_destination = {
        str(Path(str(record["destination"])).resolve(strict=False)): Path(
            str(record["bucket"])
        )
        for record in eligible
    }
    for tombstone in sorted(retired_root.glob(".deleting-*"), key=lambda item: item.name):
        source = source_by_destination.get(str(tombstone.resolve(strict=False)))
        if source is not None and (source.exists() or source.is_symlink()):
            raise StagingRetirementError(
                "staging source and retirement destination both exist: "
                + str(source)
                + " and "
                + str(tombstone)
            )
        warning = _remove_tombstone(tombstone)
        if warning is not None:
            result["warnings"].append(warning)

    for planned in eligible:
        bucket = Path(str(planned["bucket"]))
        identity = _bucket_identity(bucket)
        if identity is None:
            raise StagingRetirementError(
                "retirement plan contains a non-canonical bucket: " + str(bucket)
            )
        refreshed = classify_completed_staging_buckets(
            campaign,
            through_version=int(planned["iteration"]),
        )
        matching = [
            record
            for record in refreshed["eligible"]
            if Path(str(record["bucket"])) == bucket
        ]
        if not matching:
            destination = Path(str(planned["destination"]))
            if str(planned["action"]) == "preserve" and destination.is_dir():
                continue
            if str(planned["action"]) == "delete" and not bucket.exists():
                warning = _remove_tombstone(destination)
                if warning is not None:
                    result["warnings"].append(warning)
                continue
            raise StagingRetirementError(
                "staging bucket changed after retirement classification: " + str(bucket)
            )
        current = matching[0]
        for key in ("transaction_id", "action", "destination"):
            if str(current[key]) != str(planned[key]):
                raise StagingRetirementError(
                    "staging retirement classification changed before mutation"
                )
        destination = Path(str(current["destination"]))
        if destination.exists() or destination.is_symlink():
            raise StagingRetirementError(
                "staging source and retirement destination both exist: "
                + str(bucket)
                + " and "
                + str(destination)
            )
        if bucket.stat().st_dev != retired_root.stat().st_dev:
            raise OSError(
                "staging retirement requires source and destination on one filesystem"
            )
        os.replace(str(bucket), str(destination))
        _fsync_parent_dir(bucket)
        _fsync_parent_dir(destination)
        result["retired"].append(str(bucket))
        result["n_retired"] += 1
        if str(current["action"]) == "preserve":
            try:
                _make_owner_writable_tree(destination)
            except Exception as exc:
                result["warnings"].append(
                    "could not make retained diagnostic staging owner-writable "
                    + str(destination)
                    + ": "
                    + type(exc).__name__
                    + ": "
                    + str(exc)
                )
            result["preserved"].append(str(destination))
            result["n_preserved"] += 1
            continue
        result["deleted"].append(str(bucket))
        result["n_deleted"] += 1
        warning = _remove_tombstone(destination)
        if warning is not None:
            result["warnings"].append(warning)
    return result


def retained_staging_inventory(campaign_dir: Path) -> List[str]:
    """Return bounded retained diagnostic roots for verbose presentation."""
    root = retired_staging_root(Path(campaign_dir))
    if not root.is_dir() or root.is_symlink():
        return []
    return [
        str(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if not path.name.startswith(".deleting-")
    ]


__all__ = [
    "STAGING_RETIRED_DIRNAME",
    "StagingRetirementError",
    "classify_completed_staging_buckets",
    "retire_completed_staging_buckets",
    "retained_staging_inventory",
    "retired_staging_root",
]
