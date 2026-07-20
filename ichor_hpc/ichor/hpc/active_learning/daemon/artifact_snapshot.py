"""One-pass committed-artefact verification for recovery and status checks."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, TextIO, Tuple

from ..strict_json import strict_json as json
from ..versioning.manifest import (
    MANIFEST_FILENAME,
    read_manifest,
    sha256_file,
)
from ..versioning.reference_data import (
    PROVENANCE_FILENAME,
    REFERENCE_DATA_VERSION_FILENAME,
    ReferenceDataView,
    ReferenceDataVersioning,
)
from ..versioning.trained_models import (
    TRAINED_MODEL_SET_FILENAME,
    TrainedModelSet,
    TrainedModelVersioning,
    resolve_trained_model_chain,
)
from .filesystem import campaign_owned_path


VERIFICATION_LEVELS = frozenset({"authority", "metadata", "deep"})


class ArtefactSnapshotError(RuntimeError):
    """Raised when a committed-artefact snapshot cannot be trusted."""


class _DigestTracker:
    def __init__(
        self,
        *,
        level: str,
        progress_stream: Optional[TextIO],
        total_pointdirs: int,
    ) -> None:
        self.level = str(level)
        self.progress_stream = progress_stream
        self.total_pointdirs = int(total_pointdirs)
        self.cache: Dict[str, Tuple[str, int, int]] = {}
        self.payload_paths: set[str] = set()
        self.pointdirs_seen: set[str] = set()
        self.payload_bytes_hashed = 0
        self._last_report_bytes = 0
        self._last_report_monotonic = time.monotonic()
        self._started_monotonic = self._last_report_monotonic
        self._current_payload = False

    @staticmethod
    def _key(path: Path) -> str:
        return os.path.abspath(os.fspath(path))

    @staticmethod
    def _pointdir(path: Path) -> Optional[str]:
        for parent in (path, *path.parents):
            if parent.name.startswith("POINT_") and parent.name.endswith(".pointdir"):
                return str(parent)
        return None

    def _on_bytes(self, path: Path, size: int) -> None:
        if self._current_payload:
            self.payload_bytes_hashed += int(size)
            pointdir = self._pointdir(path)
            if pointdir is not None:
                self.pointdirs_seen.add(pointdir)
        if self.progress_stream is None or self.level != "deep":
            return
        now = time.monotonic()
        byte_delta = self.payload_bytes_hashed - self._last_report_bytes
        if now - self._last_report_monotonic < 5.0 and byte_delta < 256 * 1024 * 1024:
            return
        elapsed = max(now - self._started_monotonic, 1.0e-9)
        rate = self.payload_bytes_hashed / elapsed / (1024.0 * 1024.0)
        point_text = str(len(self.pointdirs_seen))
        if self.total_pointdirs:
            point_text += "/" + str(self.total_pointdirs)
        print(
            "Deep verification: "
            + point_text
            + " pointdirs, "
            + f"{self.payload_bytes_hashed / (1024.0 ** 3):.2f} GiB, "
            + f"{rate:.1f} MiB/s, elapsed {int(elapsed):d}s",
            file=self.progress_stream,
            flush=True,
        )
        self._last_report_monotonic = now
        self._last_report_bytes = self.payload_bytes_hashed

    def digest(self, path: Path, payload: bool) -> str:
        source = Path(path)
        key = self._key(source)
        cached = self.cache.get(key)
        if cached is not None:
            digest, expected_size, expected_mtime_ns = cached
            if source.is_symlink() or not source.is_file():
                raise ArtefactSnapshotError(
                    "cached verification target is no longer a regular file: "
                    + str(source)
                )
            current = source.stat()
            if (
                int(current.st_size) != expected_size
                or int(current.st_mtime_ns) != expected_mtime_ns
            ):
                raise ArtefactSnapshotError(
                    "verification target changed after hashing: " + str(source)
                )
            if payload and key not in self.payload_paths:
                self.payload_paths.add(key)
                self._current_payload = True
                try:
                    self._on_bytes(source, expected_size)
                finally:
                    self._current_payload = False
            return digest
        if source.is_symlink() or not source.is_file():
            raise ArtefactSnapshotError(
                "verification target is missing, non-regular or symlinked: " + str(source)
            )
        before = source.stat()
        self._current_payload = bool(payload)
        try:
            digest = sha256_file(source, progress_callback=self._on_bytes)
        finally:
            self._current_payload = False
        after = source.stat()
        if (
            int(before.st_size) != int(after.st_size)
            or int(before.st_mtime_ns) != int(after.st_mtime_ns)
        ):
            raise ArtefactSnapshotError(
                "verification target changed while hashing: " + str(source)
            )
        self.cache[key] = (
            digest,
            int(after.st_size),
            int(after.st_mtime_ns),
        )
        if payload:
            self.payload_paths.add(key)
        return digest

    @property
    def payload_files_hashed(self) -> int:
        return len(self.payload_paths)


@dataclass(frozen=True)
class CommittedArtifactSnapshot:
    verification_level: str
    committed_reference_data_versions: Tuple[int, ...]
    committed_model_versions: Tuple[int, ...]
    reference_views: Tuple[ReferenceDataView, ...]
    model_sets: Tuple[TrainedModelSet, ...]
    reference_errors: Mapping[int, str]
    model_errors: Mapping[int, str]
    campaign_uids: Tuple[str, ...]
    anchor_sha256: str
    anchor_records: Tuple[Tuple[str, int, str], ...] = field(repr=False)
    files_inspected: int = 0
    payload_files_hashed: int = 0
    payload_bytes_hashed: int = 0
    elapsed_seconds: float = 0.0
    submission_intents: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    submission_intent_errors: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    completion_receipts: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    completion_receipt_errors: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    reconcile_transactions: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    reference_commit_transactions: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    aimall_quality_revalidations: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple,
        repr=False,
    )

    @property
    def valid_reference_data_versions(self) -> Tuple[int, ...]:
        return tuple(int(view.version) for view in self.reference_views)

    @property
    def valid_model_versions(self) -> Tuple[int, ...]:
        return tuple(int(model.version) for model in self.model_sets)

    @property
    def first_invalid_version(self) -> Optional[Dict[str, Any]]:
        candidates = [
            ("reference_data", int(version), str(error))
            for version, error in self.reference_errors.items()
        ] + [
            ("models", int(version), str(error))
            for version, error in self.model_errors.items()
        ]
        if not candidates:
            return None
        kind, version, error = sorted(candidates, key=lambda item: (item[1], item[0]))[0]
        return {"kind": kind, "version": version, "error": error}

    def reference_view(self, version: int) -> ReferenceDataView:
        for view in self.reference_views:
            if int(view.version) == int(version):
                return view
        raise ArtefactSnapshotError(
            "reference-data version is not valid in the artefact snapshot: "
            + str(int(version))
        )

    def model_set(self, version: int) -> TrainedModelSet:
        for model in self.model_sets:
            if int(model.version) == int(version):
                return model
        raise ArtefactSnapshotError(
            "model version is not valid in the artefact snapshot: " + str(int(version))
        )

    def assert_anchors_unchanged(self, campaign_dir: Path) -> None:
        campaign = Path(campaign_dir)
        references = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
        models = TrainedModelVersioning(campaign / "TRAINED_MODELS")
        if _committed_versions(references) != self.committed_reference_data_versions:
            raise ArtefactSnapshotError(
                "committed reference-data inventory changed after reconcile inspection"
            )
        if _committed_versions(models) != self.committed_model_versions:
            raise ArtefactSnapshotError(
                "committed model inventory changed after reconcile inspection"
            )
        recorded_paths = tuple(record[0] for record in self.anchor_records)
        control_roots = (
            ".DATA/ACTIVE_LEARNING/submission_intents",
            ".DATA/ACTIVE_LEARNING/phase_completions",
            ".DATA/ACTIVE_LEARNING/reconcile_transactions",
            ".DATA/ACTIVE_LEARNING/reference_commit_transactions",
        )
        current_control_paths = set()
        for relative_root in control_roots:
            root = campaign / relative_root
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                raise ArtefactSnapshotError(
                    "authority control root changed after reconcile inspection: "
                    + str(root)
                )
            current_control_paths.update(
                candidate.relative_to(campaign).as_posix()
                for candidate in sorted(root.glob("*.json"))
                if candidate.is_file() and not candidate.is_symlink()
            )
        recorded_control_paths = {
            path
            for path in recorded_paths
            if any(path.startswith(root + "/") for root in control_roots)
        }
        if current_control_paths != recorded_control_paths:
            raise ArtefactSnapshotError(
                "committed artefact authority inventory changed after reconcile inspection"
            )
        current_revalidations = set(_quality_revalidation_anchor_paths(campaign))
        recorded_revalidations = {
            path
            for path in recorded_paths
            if "/allocation/quality_revalidations/" in path
        }
        if current_revalidations != recorded_revalidations:
            raise ArtefactSnapshotError(
                "AIMAll quality-revalidation inventory changed after reconcile inspection"
            )
        observed = _anchor_records(campaign_dir, recorded_paths)
        digest = _anchor_digest(observed)
        if observed != self.anchor_records or digest != self.anchor_sha256:
            raise ArtefactSnapshotError(
                "committed artefact metadata changed after reconcile inspection"
            )

    def verification_payload(self, *, deep_required: bool = False) -> Dict[str, Any]:
        return {
            "level": self.verification_level,
            "deep_required": bool(deep_required),
            "recursive_scan": self.verification_level != "authority",
            "payload_hashing": self.verification_level == "deep",
            "control_files_checked": int(len(self.anchor_records)),
            "files_inspected": int(self.files_inspected),
            "payload_files_hashed": int(self.payload_files_hashed),
            "payload_bytes_hashed": int(self.payload_bytes_hashed),
            "elapsed_seconds": float(self.elapsed_seconds),
            "anchor_sha256": self.anchor_sha256,
            "first_invalid_version": self.first_invalid_version,
        }


def _committed_versions(versioning: Any) -> Tuple[int, ...]:
    return tuple(int(value) for value in versioning.list_committed_versions())


def _first_chain_gap(versions: Sequence[int]) -> Optional[int]:
    for expected, observed in enumerate(versions):
        if int(observed) != expected:
            return expected
    return None


def _total_pointdirs(reference_root: Path, versions: Sequence[int]) -> int:
    total = 0
    versioning = ReferenceDataVersioning(reference_root)
    for version in versions:
        root = versioning.iteration_path(int(version))
        if root.is_dir() and not root.is_symlink():
            total += sum(
                1
                for path in root.glob("POINT_*.pointdir")
                if path.is_dir() and not path.is_symlink()
            )
    return total


def _authoritative_anchor_paths(
    campaign: Path,
    reference_views: Sequence[ReferenceDataView],
    model_sets: Sequence[TrainedModelSet],
    *,
    verification_level: str,
) -> Tuple[str, ...]:
    paths: set[str] = set()
    for view in reference_views:
        version_root = ReferenceDataVersioning(
            campaign / "QM_REFERENCE_DATA"
        ).iteration_path(int(view.version))
        for name in (
            MANIFEST_FILENAME,
            REFERENCE_DATA_VERSION_FILENAME,
            "REFERENCE_COMMIT_RECEIPT.json",
        ):
            paths.add((version_root / name).relative_to(campaign).as_posix())
        if verification_level != "authority":
            for relative_name in read_manifest(version_root):
                if relative_name.endswith(".json"):
                    paths.add(
                        (version_root / relative_name).relative_to(campaign).as_posix()
                    )
            for entry in view.entries:
                if int(entry.introduced_in_version) != int(view.version):
                    continue
                paths.add(
                    (entry.pointdir_path / "QUANTUM_ACCEPTANCE_RECEIPT.json")
                    .relative_to(campaign)
                    .as_posix()
                )
                paths.add(
                    (entry.pointdir_path / PROVENANCE_FILENAME)
                    .relative_to(campaign)
                    .as_posix()
                )
    for model in model_sets:
        for name in (MANIFEST_FILENAME, TRAINED_MODEL_SET_FILENAME):
            paths.add((model.root / name).relative_to(campaign).as_posix())
        if verification_level != "authority":
            for relative_name in read_manifest(model.root):
                if relative_name.endswith(".json"):
                    paths.add((model.root / relative_name).relative_to(campaign).as_posix())
    for pointer in (
        campaign / "QM_REFERENCE_DATA" / "current",
        campaign / "TRAINED_MODELS" / "current",
    ):
        if pointer.is_file() and not pointer.is_symlink():
            paths.add(pointer.relative_to(campaign).as_posix())
    for relative_name in (
        ".DATA/ACTIVE_LEARNING/state.json",
        ".DATA/ACTIVE_LEARNING/config_lock.json",
        ".DATA/ACTIVE_LEARNING/execution_identity.json",
        ".DATA/ACTIVE_LEARNING/environment_current.json",
    ):
        candidate = campaign / relative_name
        if candidate.is_file() and not candidate.is_symlink():
            paths.add(relative_name)
    environment_current = campaign / ".DATA/ACTIVE_LEARNING/environment_current.json"
    if environment_current.is_file() and not environment_current.is_symlink():
        try:
            current_payload = json.loads(
                environment_current.read_text(encoding="utf-8"),
                source=environment_current,
            )
            generation = int(current_payload["generation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtefactSnapshotError(
                "active environment-generation pointer is invalid"
            ) from exc
        generation_path = (
            campaign
            / ".DATA/ACTIVE_LEARNING/environment_generations"
            / ("generation-" + str(generation).zfill(6) + ".json")
        )
        if not generation_path.is_file() or generation_path.is_symlink():
            raise ArtefactSnapshotError(
                "active environment-generation record is missing"
            )
        paths.add(generation_path.relative_to(campaign).as_posix())
    for relative_root in (
        ".DATA/ACTIVE_LEARNING/submission_intents",
        ".DATA/ACTIVE_LEARNING/phase_completions",
        ".DATA/ACTIVE_LEARNING/reconcile_transactions",
        ".DATA/ACTIVE_LEARNING/reference_commit_transactions",
    ):
        root = campaign / relative_root
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ArtefactSnapshotError(
                "authority control root is unsafe: " + str(root)
            )
        for candidate in sorted(root.glob("*.json")):
            if candidate.is_symlink() or not candidate.is_file():
                raise ArtefactSnapshotError(
                    "authority control entry is unsafe: " + str(candidate)
                )
            paths.add(candidate.relative_to(campaign).as_posix())
    paths.update(_quality_revalidation_anchor_paths(campaign))
    return tuple(sorted(paths))


def _quality_revalidation_anchor_paths(campaign: Path) -> Tuple[str, ...]:
    from .aimall_quality_revalidation import (
        inventory_aimall_quality_revalidations,
    )

    inventory = inventory_aimall_quality_revalidations(campaign)
    errors = list(inventory.get("errors") or [])
    if errors:
        raise ArtefactSnapshotError(
            "AIMAll quality-revalidation evidence is invalid: "
            + str(errors[0].get("path") or "unknown")
            + ": "
            + str(errors[0].get("error") or "invalid ledger")
        )
    paths = set()
    for record in inventory.get("records", []):
        paths.add(str(record["path"]))
        payload = record.get("payload") or {}
        binding = payload.get("corrected_quality_manifest")
        if isinstance(binding, Mapping) and binding.get("path"):
            quality_path = campaign_owned_path(
                campaign,
                campaign / str(binding["path"]),
            )
            if quality_path.is_symlink() or not quality_path.is_file():
                raise ArtefactSnapshotError(
                    "AIMAll corrected quality evidence is missing"
                )
            if sha256_file(quality_path) != str(binding.get("sha256") or ""):
                raise ArtefactSnapshotError(
                    "AIMAll corrected quality evidence has changed"
                )
            paths.add(quality_path.relative_to(campaign).as_posix())
    return tuple(sorted(paths))


def _anchor_records(
    campaign_dir: Path,
    relative_paths: Iterable[str],
) -> Tuple[Tuple[str, int, str], ...]:
    campaign = Path(campaign_dir).resolve()
    records = []
    for relative_text in sorted(set(str(value) for value in relative_paths)):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtefactSnapshotError("authoritative anchor path is unsafe: " + relative_text)
        path = campaign / relative
        if path.is_symlink() or not path.is_file():
            raise ArtefactSnapshotError("authoritative anchor is missing: " + str(path))
        try:
            path.resolve().relative_to(campaign)
        except ValueError as exc:
            raise ArtefactSnapshotError("authoritative anchor escapes campaign") from exc
        records.append((relative.as_posix(), int(path.stat().st_size), sha256_file(path)))
    return tuple(records)


def _anchor_digest(records: Sequence[Tuple[str, int, str]]) -> str:
    encoded = json.dumps(
        [list(record) for record in records],
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _manifest_file_count(root: Path, versions: Sequence[int], versioning: Any) -> int:
    total = 0
    for version in versions:
        version_root = versioning.iteration_path(int(version))
        try:
            total += len(read_manifest(version_root))
        except Exception:
            continue
    return total


def build_committed_artifact_snapshot(
    campaign_dir: Path,
    *,
    verification_level: str = "authority",
    progress_stream: Optional[TextIO] = None,
) -> CommittedArtifactSnapshot:
    level = str(verification_level)
    if level not in VERIFICATION_LEVELS:
        raise ValueError("artefact verification must be authority, metadata or deep")
    campaign = Path(campaign_dir).resolve()
    reference_versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    model_versioning = TrainedModelVersioning(campaign / "TRAINED_MODELS")
    committed_references = _committed_versions(reference_versioning)
    committed_models = _committed_versions(model_versioning)
    tracker = _DigestTracker(
        level=level,
        progress_stream=progress_stream,
        total_pointdirs=(
            _total_pointdirs(campaign / "QM_REFERENCE_DATA", committed_references)
            if level == "deep"
            else 0
        ),
    )
    started = time.monotonic()
    reference_views: Tuple[ReferenceDataView, ...] = ()
    model_sets: Tuple[TrainedModelSet, ...] = ()
    reference_errors: Dict[int, str] = {}
    model_errors: Dict[int, str] = {}
    submission_intents: Tuple[Mapping[str, Any], ...] = ()
    submission_intent_errors: Tuple[Mapping[str, Any], ...] = ()
    completion_receipts: Tuple[Mapping[str, Any], ...] = ()
    completion_receipt_errors: Tuple[Mapping[str, Any], ...] = ()
    reconcile_transactions: Tuple[Mapping[str, Any], ...] = ()
    reference_commit_transactions: Tuple[Mapping[str, Any], ...] = ()
    aimall_quality_revalidations: Tuple[Mapping[str, Any], ...] = ()

    from .reconcile_transaction import inventory_reconcile_transactions
    from .reference_commit import inventory_reference_commits
    from .submission_intent import inventory_intents
    from .completion_receipts import inventory_completion_receipts
    from .aimall_quality_revalidation import (
        inventory_aimall_quality_revalidations,
    )

    intent_inventory = inventory_intents(campaign)
    submission_intents = tuple(
        dict(record) for record in intent_inventory.get("records", [])
    )
    submission_intent_errors = tuple(
        dict(record) for record in intent_inventory.get("errors", [])
    )
    completion_inventory = inventory_completion_receipts(campaign)
    completion_receipts = tuple(
        dict(record) for record in completion_inventory.get("records", [])
    )
    completion_receipt_errors = tuple(
        dict(record) for record in completion_inventory.get("errors", [])
    )
    reconcile_transactions = tuple(
        dict(record) for record in inventory_reconcile_transactions(campaign)
    )
    reference_commit_transactions = tuple(
        dict(record)
        for record in inventory_reference_commits(
            campaign,
            verification=("authority" if level == "authority" else "metadata"),
        )
    )
    revalidation_inventory = inventory_aimall_quality_revalidations(campaign)
    if revalidation_inventory.get("errors"):
        error = revalidation_inventory["errors"][0]
        raise ArtefactSnapshotError(
            "AIMAll quality-revalidation evidence is invalid: "
            + str(error.get("path") or "unknown")
            + ": "
            + str(error.get("error") or "invalid ledger")
        )
    aimall_quality_revalidations = tuple(
        dict(record) for record in revalidation_inventory.get("records", [])
    )

    if committed_references:
        head = max(committed_references)
        reference_gap = _first_chain_gap(committed_references)
        resolution_target = head if reference_gap is None else reference_gap - 1
        resolved_references: list[ReferenceDataView] = []
        resolution_error: Optional[Exception] = None
        if resolution_target >= 0:
            try:
                reference_views = reference_versioning.resolve_chain(
                    resolution_target,
                    verification=level,
                    digest_file=tracker.digest,
                    resolved_views_out=resolved_references,
                )
            except Exception as exc:
                resolution_error = exc
                reference_views = tuple(resolved_references)
        if resolution_error is not None or reference_gap is not None:
            first_invalid_reference = (
                len(reference_views)
                if resolution_error is not None
                else int(reference_gap)
            )
            reference_errors[first_invalid_reference] = (
                type(resolution_error).__name__ + ": " + str(resolution_error)
                if resolution_error is not None
                else "reference-data version is missing from the committed chain"
            )
            for version in committed_references:
                if int(version) > first_invalid_reference:
                    reference_errors[int(version)] = (
                        "untrusted because reference-data version "
                        + str(first_invalid_reference)
                        + " is invalid"
                    )

    if committed_models:
        head = max(committed_models)
        if len(reference_views) <= head:
            available_references = {
                int(view.version) for view in reference_views
            }
            first_missing = next(
                (
                    int(version)
                    for version in committed_models
                    if int(version) not in available_references
                ),
                int(head),
            )
            for version in committed_models:
                if int(version) >= first_missing:
                    model_errors[int(version)] = (
                        "reference-data version "
                        + str(first_missing)
                        + " is unavailable for model validation"
                    )
        else:
            model_gap = _first_chain_gap(committed_models)
            resolution_target = head if model_gap is None else model_gap - 1
            resolved_models: list[TrainedModelSet] = []
            resolution_error = None
            if resolution_target >= 0:
                try:
                    model_sets = resolve_trained_model_chain(
                        campaign,
                        resolution_target,
                        verification=level,
                        reference_views=reference_views,
                        digest_file=tracker.digest,
                        resolved_models_out=resolved_models,
                    )
                except Exception as exc:
                    resolution_error = exc
                    model_sets = tuple(resolved_models)
            if resolution_error is not None or model_gap is not None:
                first_invalid_model = (
                    len(model_sets)
                    if resolution_error is not None
                    else int(model_gap)
                )
                model_errors[first_invalid_model] = (
                    type(resolution_error).__name__ + ": " + str(resolution_error)
                    if resolution_error is not None
                    else "model version is missing from the committed chain"
                )
                for version in committed_models:
                    if int(version) > first_invalid_model:
                        model_errors[int(version)] = (
                            "untrusted because model version "
                            + str(first_invalid_model)
                            + " is invalid"
                        )

    anchor_paths = _authoritative_anchor_paths(
        campaign,
        reference_views,
        model_sets,
        verification_level=level,
    )
    anchor_records = _anchor_records(campaign, anchor_paths)
    if level == "authority":
        reference_count = len(anchor_records)
        model_count = 0
    else:
        reference_count = _manifest_file_count(
            campaign / "QM_REFERENCE_DATA",
            committed_references,
            reference_versioning,
        )
        model_count = _manifest_file_count(
            campaign / "TRAINED_MODELS",
            committed_models,
            model_versioning,
        )
    campaign_uids = tuple(
        sorted(
            {
                *(str(view.campaign_uid) for view in reference_views),
                *(str(model.campaign_uid) for model in model_sets),
            }
        )
    )
    return CommittedArtifactSnapshot(
        verification_level=level,
        committed_reference_data_versions=committed_references,
        committed_model_versions=committed_models,
        reference_views=reference_views,
        model_sets=model_sets,
        reference_errors=reference_errors,
        model_errors=model_errors,
        campaign_uids=campaign_uids,
        anchor_sha256=_anchor_digest(anchor_records),
        anchor_records=anchor_records,
        files_inspected=int(reference_count + model_count),
        payload_files_hashed=(tracker.payload_files_hashed if level == "deep" else 0),
        payload_bytes_hashed=(tracker.payload_bytes_hashed if level == "deep" else 0),
        elapsed_seconds=float(time.monotonic() - started),
        submission_intents=submission_intents,
        submission_intent_errors=submission_intent_errors,
        completion_receipts=completion_receipts,
        completion_receipt_errors=completion_receipt_errors,
        reconcile_transactions=reconcile_transactions,
        reference_commit_transactions=reference_commit_transactions,
        aimall_quality_revalidations=aimall_quality_revalidations,
    )


__all__ = [
    "ArtefactSnapshotError",
    "CommittedArtifactSnapshot",
    "VERIFICATION_LEVELS",
    "build_committed_artifact_snapshot",
]
