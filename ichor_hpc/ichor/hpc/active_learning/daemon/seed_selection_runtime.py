"""Derived SEED_SELECT caches and bounded user-visible progress."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy

from ..strict_json import strict_json as json
from .filesystem import campaign_owned_path
from .state import atomic_write_json


SEED_FEATURE_CACHE_SCHEMA_VERSION = 1
SEED_VARIANCE_CACHE_SCHEMA_VERSION = 1
SEED_NEIGHBOUR_CACHE_SCHEMA_VERSION = 1
SEED_PROJECTION_WORKSPACE_SCHEMA_VERSION = 1
SEED_MODEL_FACTOR_CACHE_SCHEMA_VERSION = 1
SEED_SELECTION_PROGRESS_STAGES = frozenset(
    {
        "trajectory_authority",
        "trajectory_coordinates",
        "model_authority",
        "models",
        "model_factors",
        "features",
        "seed_exclusions",
        "sampling_protocol",
        "reference_neighbours",
        "reference_scales",
        "filtering",
        "random",
        "variance",
        "shortlist",
        "d_optimal",
        "publishing",
        "failed",
        "complete",
    }
)
SEED_PROGRESS_SCHEMA_VERSION = 1
SEED_FEATURE_ENCODING_VERSION = 1
SEED_POSTERIOR_PROJECTION_VERSION = 1
FEATURE_CHUNK_SIZE = 256
PROGRESS_WRITE_INTERVAL_SECONDS = 2.0
PROGRESS_JOURNAL_INTERVAL_SECONDS = 30.0
MAX_RESIDENT_PROJECTION_BYTES = 256 * 1024 * 1024


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _numpy_build_sha256() -> str:
    config = getattr(np.__config__, "CONFIG", {})
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _normalise_json(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary basename no longer than ``factor.npy`` so cache
    # publication also works near the legacy Windows path-length boundary.
    temporary = path.with_name(".t" + uuid.uuid4().hex[:8])
    with temporary.open("xb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _copy_regular_file_atomic(source: Path, destination: Path) -> str:
    """Copy one immutable cache payload and return its copied SHA-256."""
    if source.is_symlink() or not source.is_file():
        raise ValueError("model-factor source is not a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + ".tmp-" + uuid.uuid4().hex[:12]
    )
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
        try:
            descriptor = os.open(str(destination.parent), os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _invalidate_directory(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    invalid = path.with_name(".invalid-" + uuid.uuid4().hex[:12])
    os.replace(path, invalid)
    shutil.rmtree(invalid, ignore_errors=True)


def finalise_seed_selection_workspace(
    campaign_dir: Path,
    *,
    iteration: int,
) -> None:
    root = campaign_owned_path(
        Path(campaign_dir), Path(".DATA") / "CACHE" / "SEED_SELECT"
    )
    workspace = (
        root
        / "workspaces"
        / ("iteration-" + str(int(iteration)).zfill(6))
    )
    _invalidate_directory(workspace)


class SeedSelectionProgressReporter:
    """Write one compact current-progress record and throttled journal events."""

    def __init__(
        self,
        campaign_dir: Path,
        *,
        campaign_uid: str,
        iteration: int,
        journal_event: Optional[Callable[..., None]] = None,
    ) -> None:
        campaign = Path(campaign_dir)
        self.path = campaign_owned_path(
            campaign,
            Path(".DATA")
            / "ACTIVE_LEARNING"
            / "runtime_progress"
            / "SEED_SELECT.json",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.campaign_uid = str(campaign_uid)
        self.iteration = int(iteration)
        self.pid = int(os.getpid())
        self.started_monotonic = time.monotonic()
        self.started_iso = datetime.now(timezone.utc).isoformat()
        launch_id = str(os.environ.get("ICHOR_DAEMON_LAUNCH_ID") or "").strip()
        self.daemon_start_identity = (
            launch_id
            if launch_id
            else "pid-" + str(self.pid) + "-" + self.started_iso
        )
        self.journal_event = journal_event
        self._input_identity: Dict[str, str] = {}
        self._last_write = 0.0
        self._last_journal = 0.0
        self._last_stage: Optional[str] = None
        self._last_payload: Dict[str, Any] = {}
        self._stage_started_monotonic = self.started_monotonic
        self._stage_initial_completed = 0
        self._journal("seed_selection_started", stage="trajectory_authority")
        self.update("trajectory_authority", force=True, completed=0, total=1)

    def bind_inputs(
        self,
        *,
        trajectory_sha256: str,
        model_set_sha256: str,
        model_manifest_sha256: str,
    ) -> None:
        """Bind the immutable scientific inputs shown in every later update."""
        self._input_identity = {
            "trajectory_sha256": str(trajectory_sha256),
            "model_set_sha256": str(model_set_sha256),
            "model_manifest_sha256": str(model_manifest_sha256),
        }

    def _journal(self, event: str, **payload: Any) -> None:
        if self.journal_event is None:
            return
        try:
            self.journal_event(
                str(event),
                phase="SEED_SELECT",
                iteration=int(self.iteration),
                **payload,
            )
        except Exception:
            return

    def update(self, stage: str, *, force: bool = False, **payload: Any) -> None:
        now = time.monotonic()
        stage_changed = str(stage) != self._last_stage
        if stage_changed and self._last_stage is not None:
            previous_elapsed = max(0.0, now - self._stage_started_monotonic)
            previous = {
                "schema_version": SEED_PROGRESS_SCHEMA_VERSION,
                "campaign_uid": self.campaign_uid,
                "phase": "SEED_SELECT",
                "iteration": int(self.iteration),
                "pid": int(self.pid),
                "daemon_start_identity": self.daemon_start_identity,
                "stage": str(self._last_stage),
                "status": "completed",
                "started_iso": self.started_iso,
                "updated_iso": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": float(now - self.started_monotonic),
                "stage_elapsed_seconds": float(previous_elapsed),
                **self._input_identity,
                **_normalise_json(self._last_payload),
            }
            try:
                atomic_write_json(self.path, previous)
            except Exception:
                pass
        if stage_changed:
            self._stage_started_monotonic = now
            try:
                self._stage_initial_completed = int(payload.get("completed", 0))
            except (TypeError, ValueError):
                self._stage_initial_completed = 0
        merged = (
            dict(payload)
            if stage_changed
            else {**self._last_payload, **payload}
        )
        stage_elapsed = max(0.0, now - self._stage_started_monotonic)
        try:
            completed_value = int(merged.get("completed", 0))
        except (TypeError, ValueError):
            completed_value = self._stage_initial_completed
        throughput = None
        if stage_elapsed > 0.0 and completed_value > self._stage_initial_completed:
            throughput = float(
                (completed_value - self._stage_initial_completed) / stage_elapsed
            )
        record = {
            "schema_version": SEED_PROGRESS_SCHEMA_VERSION,
            "campaign_uid": self.campaign_uid,
            "phase": "SEED_SELECT",
            "iteration": int(self.iteration),
            "pid": int(self.pid),
            "daemon_start_identity": self.daemon_start_identity,
            "stage": str(stage),
            "status": str(merged.pop("status", "running")),
            "started_iso": self.started_iso,
            "updated_iso": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": float(now - self.started_monotonic),
            "stage_elapsed_seconds": float(stage_elapsed),
            **self._input_identity,
            **_normalise_json(merged),
        }
        if throughput is not None and np.isfinite(throughput):
            record["throughput_per_second"] = float(throughput)
        if force or stage_changed or now - self._last_write >= PROGRESS_WRITE_INTERVAL_SECONDS:
            try:
                atomic_write_json(self.path, record)
                self._last_write = now
            except Exception:
                pass
        if stage_changed or now - self._last_journal >= PROGRESS_JOURNAL_INTERVAL_SECONDS:
            self._journal(
                "seed_selection_progress",
                stage=str(stage),
                elapsed_seconds=float(record["elapsed_seconds"]),
                stage_elapsed_seconds=float(record["stage_elapsed_seconds"]),
                selection_elapsed_seconds=float(record["elapsed_seconds"]),
                **{
                    key: record[key]
                    for key in (
                        "completed",
                        "total",
                        "eligible",
                        "shortlist_size",
                        "remaining_candidates",
                        "sample",
                        "samples",
                        "mode",
                        "modes",
                        "cache_kind",
                        "cache_status",
                        "excluded",
                        "throughput_per_second",
                    )
                    if key in record
                },
            )
            self._last_journal = now
        self._last_stage = str(stage)
        self._last_payload = merged

    def cache(self, cache_kind: str, cache_status: str, **payload: Any) -> None:
        self._journal(
            "seed_selection_cache",
            cache_kind=str(cache_kind),
            cache_status=str(cache_status),
            **_normalise_json(payload),
        )

    def callback(self, stage: str, payload: Dict[str, Any]) -> None:
        mapped = {
            "filtering": "filtering",
            "random": "random",
            "variance": "variance",
            "d_optimal": "d_optimal",
        }.get(str(stage), str(stage))
        self.update(mapped, **dict(payload))

    def finish(self, **payload: Any) -> None:
        self.update("complete", force=True, status="complete", **payload)


class CachedPoolPosterior:
    """Indexed posterior backed by immutable pool features and variances."""

    allow_prepared_legacy_fallback = True

    def __init__(
        self,
        *,
        posterior: Any,
        feature_arrays: Mapping[str, np.ndarray],
        variances: np.ndarray,
        eligible_indices: Sequence[int],
        progress: Optional[SeedSelectionProgressReporter],
        projection_directory: Path,
        feature_cache_id: str,
        variance_cache_id: str,
    ) -> None:
        self.posterior = posterior
        self.feature_arrays = {
            str(atom): np.asarray(values)
            for atom, values in feature_arrays.items()
        }
        self.eligible_indices = np.asarray(eligible_indices, dtype=np.int64)
        self.variances = np.asarray(variances, dtype=float).reshape(-1)
        if self.eligible_indices.shape != self.variances.shape:
            raise ValueError("cached posterior eligible variance count mismatch")
        if not np.all(np.isfinite(self.variances)) or np.any(self.variances < 0.0):
            raise ValueError("cached posterior variances are invalid")
        self._variance_positions = {
            int(index): position
            for position, index in enumerate(self.eligible_indices)
        }
        self._prepared = None
        self.progress = progress
        self.projection_directory = Path(projection_directory)
        self.feature_cache_id = str(feature_cache_id)
        self.variance_cache_id = str(variance_cache_id)

    @property
    def variance_population_size(self) -> int:
        return int(self.eligible_indices.size)

    @property
    def prepared_batch_available(self) -> bool:
        return self._prepared is not None

    def variances_by_index(self, indices: Sequence[int]) -> np.ndarray:
        try:
            positions = [self._variance_positions[int(index)] for index in indices]
        except KeyError as exc:
            raise KeyError("requested seed variance is outside the cached eligibility set") from exc
        return np.asarray(self.variances[positions], dtype=float)

    def prepare_indexed_batch(self, indices: Sequence[int]):
        unique = list(dict.fromkeys(int(index) for index in indices))
        if self._prepared is not None:
            available = set(int(value) for value in self._prepared.row_ids)
            if set(unique) <= available:
                return self._prepared
        if self.progress is not None:
            self.progress.update(
                "shortlist",
                force=True,
                completed=0,
                total=int(len(unique)),
                shortlist_size=int(len(unique)),
            )
        arrays = {
            atom: np.asarray(values[unique], dtype=float)
            for atom, values in self.feature_arrays.items()
        }
        estimated_projection_bytes = sum(
            int(model.ntrain) * int(len(unique)) * np.dtype(np.float64).itemsize
            for model in self.posterior._property_models.values()
        )
        use_workspace = estimated_projection_bytes > MAX_RESIDENT_PROJECTION_BYTES
        identity = self._projection_identity(unique)
        workspace = self.projection_directory.parent
        projection_ledger = None
        projection_resume: Dict[str, int] = {}
        if use_workspace and (workspace.exists() or workspace.is_symlink()):
            prepared_path = workspace / "PREPARED.json"
            if prepared_path.is_file() and not prepared_path.is_symlink():
                try:
                    self._prepared = self._load_projection_workspace(
                        workspace,
                        identity=identity,
                        row_ids=unique,
                        feature_arrays=arrays,
                    )
                    if self.progress is not None:
                        self.progress.cache(
                            "shortlist_projections",
                            "hit",
                            workspace_id=str(identity["workspace_id"]),
                        )
                        self.progress.update(
                            "shortlist",
                            force=True,
                            completed=int(len(unique)),
                            total=int(len(unique)),
                            shortlist_size=int(len(unique)),
                            projection_bytes=int(estimated_projection_bytes),
                            projection_storage="memory_map",
                            cache_status="reused",
                        )
                    return self._prepared
                except Exception:
                    _invalidate_directory(workspace)
            else:
                try:
                    projection_ledger, projection_resume = (
                        self._prepare_projection_build(
                            workspace,
                            identity=identity,
                            row_count=int(len(unique)),
                        )
                    )
                except Exception:
                    _invalidate_directory(workspace)
            if not workspace.exists() and not workspace.is_symlink():
                if self.progress is not None:
                    self.progress.cache(
                        "shortlist_projections",
                        "invalid_rebuild",
                        workspace_id=str(identity["workspace_id"]),
                    )
        elif workspace.exists() or workspace.is_symlink():
            _invalidate_directory(workspace)
        if use_workspace and projection_ledger is None:
            projection_ledger, projection_resume = self._prepare_projection_build(
                workspace,
                identity=identity,
                row_count=int(len(unique)),
            )

        def _projection_progress(atom: str, start: int, stop: int) -> None:
            assert projection_ledger is not None
            self._projection_progress_callback(
                workspace,
                projection_ledger,
                atom=str(atom),
                start=int(start),
                stop=int(stop),
            )

        self._prepared = self.posterior.prepare_feature_batch(
            arrays,
            row_ids=unique,
            projection_directory=self.projection_directory,
            max_resident_projection_bytes=MAX_RESIDENT_PROJECTION_BYTES,
            projection_resume_columns=(projection_resume if use_workspace else None),
            projection_progress_callback=(
                _projection_progress if use_workspace else None
            ),
        )
        if use_workspace:
            self._publish_projection_workspace(
                workspace,
                identity=identity,
                prepared=self._prepared,
            )
            (workspace / "BUILD.json").unlink(missing_ok=True)
            if self.progress is not None:
                self.progress.cache(
                    "shortlist_projections",
                    "published",
                    workspace_id=str(identity["workspace_id"]),
                )
        if self.progress is not None:
            self.progress.update(
                "shortlist",
                force=True,
                completed=int(len(unique)),
                total=int(len(unique)),
                shortlist_size=int(len(unique)),
                projection_bytes=int(estimated_projection_bytes),
                projection_storage=(
                    "memory"
                    if estimated_projection_bytes <= MAX_RESIDENT_PROJECTION_BYTES
                    else "memory_map"
                ),
            )
        return self._prepared

    def _projection_identity(self, row_ids: Sequence[int]) -> Dict[str, Any]:
        ids = np.asarray(row_ids, dtype="<i8")
        payload = {
            "schema_version": SEED_PROJECTION_WORKSPACE_SCHEMA_VERSION,
            "projection_version": SEED_POSTERIOR_PROJECTION_VERSION,
            "feature_cache_id": self.feature_cache_id,
            "variance_cache_id": self.variance_cache_id,
            "row_count": int(ids.size),
            "row_ids_sha256": hashlib.sha256(ids.tobytes(order="C")).hexdigest(),
            "property_name": str(self.posterior.property_name),
            "scaled": bool(self.posterior.scaled),
            "atom_models": [
                {
                    "atom": str(atom),
                    "ntrain": int(model.ntrain),
                    "nfeats": int(model.nfeats),
                }
                for atom, model in self.posterior._property_models.items()
            ],
            "dtype": "float64",
        }
        return {**payload, "workspace_id": _canonical_sha256(payload)}

    def _projection_build_records(self, row_count: int) -> Dict[str, Any]:
        return {
            str(atom): {
                "path": (
                    "projections/projection-"
                    + str(position).zfill(4)
                    + ".npy"
                ),
                "shape": [int(model.ntrain), int(row_count)],
                "completed_columns": 0,
                "chunks": [],
            }
            for position, (atom, model) in enumerate(
                self.posterior._property_models.items()
            )
        }

    @staticmethod
    def _projection_chunk_sha(
        values: np.ndarray,
        start: int,
        stop: int,
    ) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(
                values[:, int(start) : int(stop)], dtype="<f8"
            ).tobytes(order="C")
        ).hexdigest()

    def _prepare_projection_build(
        self,
        workspace: Path,
        *,
        identity: Mapping[str, Any],
        row_count: int,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        build_path = workspace / "BUILD.json"
        expected_records = self._projection_build_records(int(row_count))
        if not workspace.exists() and not workspace.is_symlink():
            self.projection_directory.mkdir(parents=True, exist_ok=False)
            ledger = {
                "schema_version": SEED_PROJECTION_WORKSPACE_SCHEMA_VERSION,
                "identity": dict(identity),
                "atoms": expected_records,
            }
            atomic_write_json(build_path, ledger)
            return ledger, {atom: 0 for atom in expected_records}
        if workspace.is_symlink() or not workspace.is_dir() or build_path.is_symlink():
            raise ValueError("seed projection build path is invalid")
        ledger = json.loads(build_path.read_text(encoding="utf-8"))
        if (
            ledger.get("schema_version")
            != SEED_PROJECTION_WORKSPACE_SCHEMA_VERSION
            or ledger.get("identity") != dict(identity)
        ):
            raise ValueError("seed projection build identity mismatch")
        records = ledger.get("atoms")
        if not isinstance(records, dict) or set(records) != set(expected_records):
            raise ValueError("seed projection build atom inventory mismatch")
        resume: Dict[str, int] = {}
        for atom, expected in expected_records.items():
            record = records.get(atom)
            if not isinstance(record, dict):
                raise ValueError("seed projection build atom record is invalid")
            if (
                record.get("path") != expected["path"]
                or record.get("shape") != expected["shape"]
            ):
                raise ValueError("seed projection build array identity mismatch")
            path = workspace / str(expected["path"])
            chunks = list(record.get("chunks") or [])
            valid_chunks = []
            expected_start = 0
            array = None
            if chunks:
                if path.is_symlink() or not path.is_file():
                    raise ValueError("seed projection build array is missing")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if (
                    list(array.shape) != list(expected["shape"])
                    or array.dtype != np.dtype(np.float64)
                ):
                    raise ValueError("seed projection build array contract mismatch")
            try:
                for chunk in chunks:
                    start = int(chunk["start"])
                    stop = int(chunk["stop"])
                    if (
                        start != expected_start
                        or stop <= start
                        or stop > int(row_count)
                    ):
                        break
                    assert array is not None
                    if self._projection_chunk_sha(array, start, stop) != str(
                        chunk.get("sha256") or ""
                    ):
                        break
                    valid_chunks.append(dict(chunk))
                    expected_start = int(stop)
            finally:
                if array is not None:
                    memory_map = getattr(array, "_mmap", None)
                    if memory_map is not None:
                        memory_map.close()
            record["chunks"] = valid_chunks
            record["completed_columns"] = int(expected_start)
            resume[str(atom)] = int(expected_start)
        atomic_write_json(build_path, ledger)
        if self.progress is not None:
            self.progress.cache(
                "shortlist_projections",
                "resumed",
                workspace_id=str(identity["workspace_id"]),
                completed_columns=int(sum(resume.values())),
            )
        return ledger, resume

    def _projection_progress_callback(
        self,
        workspace: Path,
        ledger: Dict[str, Any],
        *,
        atom: str,
        start: int,
        stop: int,
    ) -> None:
        records = ledger["atoms"]
        record = records[str(atom)]
        if int(record.get("completed_columns", 0)) != int(start):
            raise ValueError("seed projection build progress is not contiguous")
        path = workspace / str(record["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError("seed projection build array is missing")
        _fsync_file(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            digest = self._projection_chunk_sha(array, int(start), int(stop))
        finally:
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        record["chunks"] = list(record.get("chunks") or []) + [
            {
                "start": int(start),
                "stop": int(stop),
                "sha256": digest,
            }
        ]
        record["completed_columns"] = int(stop)
        atomic_write_json(workspace / "BUILD.json", ledger)
        if self.progress is not None:
            total = int(
                sum(int(item["shape"][1]) for item in records.values())
            )
            completed = int(
                sum(
                    int(item.get("completed_columns", 0))
                    for item in records.values()
                )
            )
            self.progress.update(
                "shortlist",
                completed=completed,
                total=total,
                shortlist_size=int(record["shape"][1]),
                cache_kind="shortlist_projections",
                cache_status="building",
            )

    def _load_projection_workspace(
        self,
        workspace: Path,
        *,
        identity: Mapping[str, Any],
        row_ids: Sequence[int],
        feature_arrays: Mapping[str, np.ndarray],
    ):
        from ichor.core.adversarial.posterior import (
            PreparedTotalEnergyPosteriorBatch,
        )

        manifest_path = workspace / "PREPARED.json"
        if workspace.is_symlink() or manifest_path.is_symlink():
            raise ValueError("seed projection workspace is symlinked")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("identity") != dict(identity):
            raise ValueError("seed projection workspace identity mismatch")
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("seed projection workspace inventory is invalid")

        def _load_file(record: Any, *, mmap: bool = True) -> np.ndarray:
            if not isinstance(record, dict):
                raise ValueError("seed projection workspace file record is invalid")
            path = workspace / str(record.get("path") or "")
            if path.is_symlink() or not path.is_file():
                raise ValueError("seed projection workspace file is missing")
            if path.stat().st_size != int(record.get("size", -1)):
                raise ValueError("seed projection workspace file size mismatch")
            if _sha256_file(path) != str(record.get("sha256") or ""):
                raise ValueError("seed projection workspace file hash mismatch")
            return np.load(
                path,
                mmap_mode="r" if mmap else None,
                allow_pickle=False,
            )

        means = _load_file(files.get("means"), mmap=False)
        variances = _load_file(files.get("variances"), mmap=False)
        expected_values = (int(len(row_ids)),)
        if (
            means.shape != expected_values
            or variances.shape != expected_values
            or means.dtype != np.dtype(np.float64)
            or variances.dtype != np.dtype(np.float64)
        ):
            raise ValueError("seed projection workspace values are invalid")
        projections: Dict[str, np.ndarray] = {}
        projection_records = files.get("projections")
        if not isinstance(projection_records, dict):
            raise ValueError("seed projection workspace projections are invalid")
        for atom, model in self.posterior._property_models.items():
            projection = _load_file(projection_records.get(str(atom)))
            if projection.shape != (int(model.ntrain), int(len(row_ids))):
                raise ValueError("seed projection workspace shape mismatch")
            if projection.dtype != np.dtype(np.float64):
                raise ValueError("seed projection workspace dtype mismatch")
            projections[str(atom)] = projection
        if set(projection_records) != set(projections):
            raise ValueError("seed projection workspace atom coverage mismatch")
        signal_variances = payload.get("signal_variances")
        if not isinstance(signal_variances, dict):
            raise ValueError("seed projection workspace scales are invalid")
        return PreparedTotalEnergyPosteriorBatch(
            posterior=self.posterior,
            row_ids=np.asarray(row_ids, dtype=np.int64),
            features=feature_arrays,
            projections=projections,
            means=means,
            variances=variances,
            signal_variances=signal_variances,
        )

    def _publish_projection_workspace(
        self,
        workspace: Path,
        *,
        identity: Mapping[str, Any],
        prepared: Any,
    ) -> None:
        if workspace.is_symlink() or not self.projection_directory.is_dir():
            raise ValueError("seed projection workspace was not prepared")
        means_path = workspace / "means.npy"
        variances_path = workspace / "variances.npy"
        _write_npy_atomic(means_path, np.asarray(prepared.means, dtype=np.float64))
        _write_npy_atomic(
            variances_path,
            np.asarray(prepared.variances, dtype=np.float64),
        )

        def _record(path: Path) -> Dict[str, Any]:
            return {
                "path": path.relative_to(workspace).as_posix(),
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }

        projections: Dict[str, Any] = {}
        for position, atom in enumerate(self.posterior._property_models):
            path = self.projection_directory / (
                "projection-" + str(position).zfill(4) + ".npy"
            )
            projections[str(atom)] = _record(path)
        payload = {
            "schema_version": SEED_PROJECTION_WORKSPACE_SCHEMA_VERSION,
            "identity": dict(identity),
            "signal_variances": {
                str(atom): float(value)
                for atom, value in prepared.signal_variances.items()
            },
            "files": {
                "means": _record(means_path),
                "variances": _record(variances_path),
                "projections": projections,
            },
        }
        atomic_write_json(workspace / "PREPARED.json", payload)

    def cross_covariances_by_index(
        self, left_indices: Sequence[int], right_indices: Sequence[int]
    ) -> np.ndarray:
        prepared = self.prepare_indexed_batch(
            list(left_indices) + list(right_indices)
        )
        return prepared.cross_covariances_by_index(left_indices, right_indices)

    def close(self) -> None:
        prepared = self._prepared
        if prepared is None:
            return
        for values in prepared.projections.values():
            memory_map = getattr(values, "_mmap", None)
            if memory_map is None:
                memory_map = getattr(getattr(values, "base", None), "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        self._prepared = None


class DeferredCachedPoolPosterior:
    """Delay posterior scoring until selection has drawn its random subset."""

    allow_prepared_legacy_fallback = True

    def __init__(
        self,
        cache: "SeedSelectionRuntimeCache",
        *,
        eligible_indices: Sequence[int],
        chunk_size: int,
    ) -> None:
        self._cache = cache
        self._eligible_indices = tuple(int(value) for value in eligible_indices)
        self._chunk_size = int(chunk_size)
        self._resolved: Optional[CachedPoolPosterior] = None

    @property
    def resolved(self) -> bool:
        return self._resolved is not None

    @property
    def variance_population_size(self) -> int:
        return int(len(self._eligible_indices))

    @property
    def prepared_batch_available(self) -> bool:
        return (
            self._resolved is not None
            and self._resolved.prepared_batch_available
        )

    def _resolve(self) -> CachedPoolPosterior:
        if self._resolved is None:
            self._resolved = self._cache._ensure_indexed_posterior_now(
                eligible_indices=self._eligible_indices,
                chunk_size=self._chunk_size,
            )
        return self._resolved

    @property
    def feature_cache_id(self) -> str:
        return self._resolve().feature_cache_id

    @property
    def variance_cache_id(self) -> str:
        return self._resolve().variance_cache_id

    def variances_by_index(self, indices: Sequence[int]) -> np.ndarray:
        return self._resolve().variances_by_index(indices)

    def prepare_indexed_batch(self, indices: Sequence[int]):
        return self._resolve().prepare_indexed_batch(indices)

    def cross_covariances_by_index(
        self, left_indices: Sequence[int], right_indices: Sequence[int]
    ) -> np.ndarray:
        return self._resolve().cross_covariances_by_index(
            left_indices,
            right_indices,
        )

    def close(self) -> None:
        if self._resolved is not None:
            self._resolved.close()


class SeedSelectionRuntimeCache:
    """Build and validate cache records for one verified pool/model context."""

    def __init__(
        self,
        campaign_dir: Path,
        *,
        pool: Any,
        posterior: Any,
        model_set_sha256: str,
        model_manifest_sha256: str,
        iteration: int,
        progress: Optional[SeedSelectionProgressReporter] = None,
        model_file_sha256_by_atom: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.pool = pool
        self.posterior = posterior
        self.model_set_sha256 = str(model_set_sha256)
        self.model_manifest_sha256 = str(model_manifest_sha256)
        self.iteration = int(iteration)
        self.progress = progress
        self.model_file_sha256_by_atom = {
            str(atom): str(digest)
            for atom, digest in (model_file_sha256_by_atom or {}).items()
        }
        self.root = campaign_owned_path(
            self.campaign_dir, Path(".DATA") / "CACHE" / "SEED_SELECT"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.projection_directory = (
            self.root
            / "workspaces"
            / ("iteration-" + str(self.iteration).zfill(6))
            / "projections"
        )
        self._active_features: Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]] = None
        self.active_neighbour_cache_id: Optional[str] = None

    def _model_factor_identity(self, atom: str, model: Any) -> Dict[str, Any]:
        file_sha = self.model_file_sha256_by_atom.get(str(atom), "")
        if len(file_sha) != 64 or any(ch not in "0123456789abcdef" for ch in file_sha):
            raise ValueError("current model file identity is unavailable for " + str(atom))
        return {
            "schema_version": SEED_MODEL_FACTOR_CACHE_SCHEMA_VERSION,
            "model_set_sha256": self.model_set_sha256,
            "model_file_sha256": file_sha,
            "numeric_model_identity": str(model.numeric_identity),
            "atom": str(atom),
            "property": str(model.prop),
            "ntrain": int(model.ntrain),
            "nfeats": int(model.nfeats),
            "jitter": float(model.jitter),
            "factor_algorithm": "numpy.linalg.cholesky:model.R:v1",
            "dtype": np.dtype(np.float64).str,
            "numpy_version": str(np.__version__),
            "scipy_version": str(scipy.__version__),
            "numpy_build_sha256": _numpy_build_sha256(),
            "machine": str(platform.machine()),
            "byteorder": sys.byteorder,
        }

    @staticmethod
    def _factor_residual_is_valid(model: Any, factor: np.ndarray) -> bool:
        ntrain = int(model.ntrain)
        if ntrain <= 0:
            return False
        probes = sorted({0, ntrain // 2, ntrain - 1})
        x = np.asarray(model.x, dtype=np.float64)
        for left in probes:
            for right in probes:
                stop = min(left, right) + 1
                observed = float(
                    np.dot(factor[left, :stop], factor[right, :stop])
                )
                expected = float(
                    np.asarray(
                        model.prior_covariance(
                            x[left : left + 1], x[right : right + 1]
                        ),
                        dtype=np.float64,
                    )[0, 0]
                )
                if left == right:
                    expected += float(model.jitter)
                tolerance = max(1.0e-12, abs(expected) * 5.0e-10)
                if not np.isfinite(observed) or abs(observed - expected) > tolerance:
                    return False
        return True

    def _read_model_factor(
        self,
        data_path: Path,
        manifest_path: Path,
        *,
        identity: Mapping[str, Any],
        model: Any,
    ) -> np.ndarray:
        if (
            data_path.is_symlink()
            or manifest_path.is_symlink()
            or not data_path.is_file()
            or not manifest_path.is_file()
        ):
            raise ValueError("model-factor cache files are missing or unsafe")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SEED_MODEL_FACTOR_CACHE_SCHEMA_VERSION:
            raise ValueError("model-factor cache schema is unsupported")
        if payload.get("identity") != dict(identity):
            raise ValueError("model-factor cache identity mismatch")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise ValueError("model-factor cache data record is invalid")
        if int(data.get("size", -1)) != int(data_path.stat().st_size):
            raise ValueError("model-factor cache size mismatch")
        if str(data.get("sha256") or "") != _sha256_file(data_path):
            raise ValueError("model-factor cache hash mismatch")
        expected_shape = (int(model.ntrain), int(model.ntrain))
        if data.get("shape") != list(expected_shape):
            raise ValueError("model-factor cache shape record mismatch")
        if data.get("dtype") != np.dtype(np.float64).str:
            raise ValueError("model-factor cache dtype record mismatch")
        factor = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if factor.shape != expected_shape or factor.dtype != np.dtype(np.float64):
            raise ValueError("model-factor cache array contract mismatch")
        if not np.all(np.isfinite(factor)):
            raise ValueError("model-factor cache contains non-finite values")
        if np.any(np.diag(factor) <= 0.0):
            raise ValueError("model-factor cache diagonal is invalid")
        scale = max(1.0, float(np.max(np.abs(factor))))
        tolerance = np.finfo(np.float64).eps * max(1, int(model.ntrain)) * scale * 16.0
        if np.any(np.abs(np.triu(factor, k=1)) > tolerance):
            raise ValueError("model-factor cache is not lower triangular")
        if not self._factor_residual_is_valid(model, factor):
            raise ValueError("model-factor cache covariance residual is invalid")
        return factor

    def adopt_model_factor(self, atom: str, factor: Any) -> str:
        """Install validated external factor evidence in the canonical cache."""
        import portalocker

        atom_name = str(atom)
        model = self.posterior._property_models.get(atom_name)
        if model is None:
            raise ValueError("posterior model is unavailable for " + atom_name)
        identity = self._model_factor_identity(atom_name, model)
        cache_id = _canonical_sha256(identity)
        namespace = (
            self.root / "model_factors" / self.model_set_sha256[:24]
        )
        namespace.mkdir(parents=True, exist_ok=True)
        path_token = cache_id[:32]
        data_path = namespace / (path_token + ".npy")
        manifest_path = namespace / (path_token + ".json")
        lock_path = namespace / (path_token + ".lock")
        if namespace.is_symlink() or lock_path.is_symlink():
            raise ValueError("model-factor cache path is symlinked")
        with portalocker.Lock(str(lock_path), mode="a", timeout=600):
            try:
                restored = self._read_model_factor(
                    data_path,
                    manifest_path,
                    identity=identity,
                    model=model,
                )
                model.install_lower_cholesky(
                    restored,
                    expected_numeric_identity=str(model.numeric_identity),
                )
                return "existing_hit"
            except Exception:
                data_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)

            source_factor = np.asarray(factor)
            expected_shape = (int(model.ntrain), int(model.ntrain))
            if (
                source_factor.dtype != np.dtype(np.float64)
                or source_factor.shape != expected_shape
                or not np.all(np.isfinite(source_factor))
                or np.any(np.diag(source_factor) <= 0.0)
            ):
                raise ValueError("adopted model-factor array is invalid")
            scale = max(1.0, float(np.max(np.abs(source_factor))))
            tolerance = (
                np.finfo(np.float64).eps
                * max(1, int(model.ntrain))
                * scale
                * 16.0
            )
            if np.any(np.abs(np.triu(source_factor, k=1)) > tolerance):
                raise ValueError("adopted model-factor is not lower triangular")
            if not self._factor_residual_is_valid(model, source_factor):
                raise ValueError("adopted model-factor residual is invalid")
            source_filename = getattr(factor, "filename", None)
            copied_sha = None
            if source_filename:
                source_path = Path(str(source_filename))
                if source_path != data_path:
                    copied_sha = _copy_regular_file_atomic(
                        source_path,
                        data_path,
                    )
            if copied_sha is None:
                _write_npy_atomic(data_path, source_factor)
                copied_sha = _sha256_file(data_path)
            elif copied_sha != _sha256_file(data_path):
                raise ValueError("copied model-factor hash mismatch")
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": SEED_MODEL_FACTOR_CACHE_SCHEMA_VERSION,
                    "identity": identity,
                    "data": {
                        "size": int(data_path.stat().st_size),
                        "sha256": copied_sha,
                        "shape": [int(model.ntrain), int(model.ntrain)],
                        "dtype": np.dtype(np.float64).str,
                    },
                },
            )
            restored = self._read_model_factor(
                data_path,
                manifest_path,
                identity=identity,
                model=model,
            )
            model.install_lower_cholesky(
                restored,
                expected_numeric_identity=str(model.numeric_identity),
            )
            return "task_adopted"

    def restore_model_factor(self, atom: str) -> bool:
        """Restore one canonical factor without computing a missing value."""
        import portalocker

        atom_name = str(atom)
        model = self.posterior._property_models.get(atom_name)
        if model is None:
            raise ValueError("posterior model is unavailable for " + atom_name)
        identity = self._model_factor_identity(atom_name, model)
        cache_id = _canonical_sha256(identity)
        namespace = self.root / "model_factors" / self.model_set_sha256[:24]
        path_token = cache_id[:32]
        data_path = namespace / (path_token + ".npy")
        manifest_path = namespace / (path_token + ".json")
        lock_path = namespace / (path_token + ".lock")
        if lock_path.is_symlink():
            raise ValueError("model-factor cache lock is symlinked")
        namespace.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(lock_path), mode="a", timeout=600):
            restored = self._read_model_factor(
                data_path,
                manifest_path,
                identity=identity,
                model=model,
            )
            model.install_lower_cholesky(
                restored,
                expected_numeric_identity=str(model.numeric_identity),
            )
        return True

    def ensure_model_factors(self) -> Dict[str, str]:
        """Restore or compute Cholesky factors for the current posterior models."""
        import portalocker

        namespace_token = self.model_set_sha256[:24]
        namespace = self.root / "model_factors" / namespace_token
        namespace_ready = True
        try:
            namespace.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError):
            namespace_ready = False
        statuses: Dict[str, str] = {}
        models = sorted(self.posterior._property_models.items())
        total = len(models)
        for position, (atom, model) in enumerate(models, start=1):
            status = "computed"
            try:
                if not namespace_ready:
                    raise OSError("model-factor cache directory is unavailable")
                identity = self._model_factor_identity(str(atom), model)
                cache_id = _canonical_sha256(identity)
                path_token = cache_id[:32]
                data_path = namespace / (path_token + ".npy")
                manifest_path = namespace / (path_token + ".json")
                lock_path = namespace / (path_token + ".lock")
                if lock_path.is_symlink():
                    raise ValueError("model-factor cache lock is symlinked")
                with portalocker.Lock(str(lock_path), mode="a", timeout=600):
                    try:
                        factor = self._read_model_factor(
                            data_path,
                            manifest_path,
                            identity=identity,
                            model=model,
                        )
                        model.install_lower_cholesky(
                            factor,
                            expected_numeric_identity=str(model.numeric_identity),
                        )
                        status = "hit"
                    except Exception:
                        data_path.unlink(missing_ok=True)
                        manifest_path.unlink(missing_ok=True)
                        factor = np.asarray(model.lower_cholesky, dtype=np.float64)
                        _write_npy_atomic(data_path, factor)
                        atomic_write_json(
                            manifest_path,
                            {
                                "schema_version": SEED_MODEL_FACTOR_CACHE_SCHEMA_VERSION,
                                "identity": identity,
                                "data": {
                                    "size": int(data_path.stat().st_size),
                                    "sha256": _sha256_file(data_path),
                                    "shape": [int(model.ntrain), int(model.ntrain)],
                                    "dtype": np.dtype(np.float64).str,
                                },
                            },
                        )
                        restored = self._read_model_factor(
                            data_path,
                            manifest_path,
                            identity=identity,
                            model=model,
                        )
                        model.install_lower_cholesky(
                            restored,
                            expected_numeric_identity=str(model.numeric_identity),
                        )
                        status = "published"
            except Exception:
                # Derived-cache failure must never replace the strict numeric path.
                _ = model.lower_cholesky
                status = "fallback"
            statuses[str(atom)] = status
            if self.progress is not None:
                self.progress.update(
                    "model_factors",
                    completed=int(position),
                    total=int(total),
                    cache_kind="model_factors",
                    cache_status=status,
                )
        if self.progress is not None:
            aggregate = "hit" if statuses and set(statuses.values()) == {"hit"} else "ready"
            self.progress.cache(
                "model_factors",
                aggregate,
                n_models=int(total),
            )
        return statuses

    def _neighbour_identity(
        self,
        *,
        anchor_frame_id: int,
        max_neighbours: int,
        deduplicate_rmsd: float,
    ) -> Dict[str, Any]:
        return {
            "schema_version": SEED_NEIGHBOUR_CACHE_SCHEMA_VERSION,
            "trajectory_sha256": str(self.pool.sha256),
            "anchor_frame_id": int(anchor_frame_id),
            "atom_types": list(self.pool.manifest.atom_types),
            "masses": [float(value) for value in self.pool.manifest.masses],
            "max_neighbours": int(max_neighbours),
            "deduplicate_rmsd": float(deduplicate_rmsd),
        }

    def load_reference_neighbours(
        self,
        *,
        anchor_frame_id: int,
        max_neighbours: int,
        deduplicate_rmsd: float,
    ):
        from ichor.core.adversarial.geometry import Neighbour

        identity = self._neighbour_identity(
            anchor_frame_id=int(anchor_frame_id),
            max_neighbours=int(max_neighbours),
            deduplicate_rmsd=float(deduplicate_rmsd),
        )
        cache_id = _canonical_sha256(identity)
        self.active_neighbour_cache_id = str(cache_id)
        path = self.root / "neighbours" / (cache_id + ".json")
        if not path.exists() and not path.is_symlink():
            if self.progress is not None:
                self.progress.cache("reference_neighbours", "miss", cache_id=cache_id)
            return None, cache_id
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("reference-neighbour cache is not a regular file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != SEED_NEIGHBOUR_CACHE_SCHEMA_VERSION:
                raise ValueError("unsupported reference-neighbour cache schema")
            if payload.get("identity") != identity:
                raise ValueError("reference-neighbour cache identity mismatch")
            records = payload.get("neighbours")
            if not isinstance(records, list) or not records:
                raise ValueError("reference-neighbour cache is empty")
            if payload.get("records_sha256") != _canonical_sha256(records):
                raise ValueError("reference-neighbour cache digest mismatch")
            neighbours = []
            seen = set()
            for record in records:
                if not isinstance(record, dict) or set(record) != {
                    "frame_id",
                    "aligned_distance",
                }:
                    raise ValueError("reference-neighbour cache record is invalid")
                frame_id = int(record["frame_id"])
                distance = float(record["aligned_distance"])
                if (
                    frame_id in seen
                    or frame_id < 0
                    or frame_id >= int(self.pool.n_frames())
                    or not np.isfinite(distance)
                    or distance < 0.0
                ):
                    raise ValueError("reference-neighbour cache value is invalid")
                seen.add(frame_id)
                neighbours.append(
                    Neighbour(
                        index=frame_id,
                        atoms=self.pool.frame(frame_id),
                        aligned_distance=distance,
                    )
                )
            if len(neighbours) > int(max_neighbours):
                raise ValueError("reference-neighbour cache exceeds configured count")
            if self.progress is not None:
                self.progress.cache("reference_neighbours", "hit", cache_id=cache_id)
            return neighbours, cache_id
        except Exception:
            path.unlink(missing_ok=True)
            if self.progress is not None:
                self.progress.cache(
                    "reference_neighbours", "invalid_rebuild", cache_id=cache_id
                )
            return None, cache_id

    def store_reference_neighbours(
        self,
        *,
        cache_id: str,
        anchor_frame_id: int,
        max_neighbours: int,
        deduplicate_rmsd: float,
        neighbours: Sequence[Any],
    ) -> Path:
        identity = self._neighbour_identity(
            anchor_frame_id=int(anchor_frame_id),
            max_neighbours=int(max_neighbours),
            deduplicate_rmsd=float(deduplicate_rmsd),
        )
        if str(cache_id) != _canonical_sha256(identity):
            raise ValueError("reference-neighbour cache identity changed")
        records = [
            {
                "frame_id": int(item.index),
                "aligned_distance": float(item.aligned_distance),
            }
            for item in neighbours
        ]
        payload = {
            "schema_version": SEED_NEIGHBOUR_CACHE_SCHEMA_VERSION,
            "cache_id": str(cache_id),
            "identity": identity,
            "neighbours": records,
            "records_sha256": _canonical_sha256(records),
        }
        path = self.root / "neighbours" / (str(cache_id) + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
        if self.progress is not None:
            self.progress.cache(
                "reference_neighbours", "published", cache_id=str(cache_id)
            )
        return path

    def _feature_identity(self) -> Dict[str, Any]:
        contract_sha = ""
        try:
            from .ferebus_row_cache import read_feature_contract

            contract_sha = str(
                read_feature_contract(self.campaign_dir)["contract_sha256"]
            )
        except Exception:
            contract_sha = _canonical_sha256(
                {
                    "atom_names": sorted(self.posterior._property_models),
                    "ialf_dict": _normalise_json(self.posterior.models.ialf_dict),
                }
            )
        return {
            "schema_version": SEED_FEATURE_CACHE_SCHEMA_VERSION,
            "feature_encoding_version": SEED_FEATURE_ENCODING_VERSION,
            "trajectory_sha256": str(self.pool.sha256),
            "n_frames": int(self.pool.n_frames()),
            "natoms": int(self.pool.manifest.natoms),
            "atom_types": list(self.pool.manifest.atom_types),
            "feature_contract_sha256": contract_sha,
            "atom_models": [
                {
                    "atom": str(atom),
                    "nfeats": int(model.nfeats),
                }
                for atom, model in sorted(self.posterior._property_models.items())
            ],
            "dtype": "float64",
            "byteorder": sys.byteorder,
            "numpy_version": str(np.__version__),
            "scipy_version": str(scipy.__version__),
            "numpy_build_sha256": _numpy_build_sha256(),
            "machine": str(platform.machine()),
        }

    @staticmethod
    def _chunk_sha(
        arrays: Mapping[str, np.ndarray], start: int, stop: int
    ) -> str:
        digest = hashlib.sha256()
        for atom in sorted(arrays):
            digest.update(str(atom).encode("utf-8"))
            digest.update(
                np.ascontiguousarray(arrays[atom][start:stop], dtype=np.float64)
                .tobytes(order="C")
            )
        return digest.hexdigest()

    def _read_feature_cache(
        self, directory: Path, identity: Mapping[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        manifest_path = directory / "SEED_FEATURE_CACHE.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("seed feature cache manifest is missing")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SEED_FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported seed feature cache schema")
        if payload.get("identity") != dict(identity):
            raise ValueError("seed feature cache identity mismatch")
        files = payload.get("files")
        if not isinstance(files, dict):
            raise ValueError("seed feature cache file inventory is invalid")
        arrays: Dict[str, np.ndarray] = {}
        for atom, record in files.items():
            if not isinstance(record, dict):
                raise ValueError("seed feature cache file record is invalid")
            path = directory / str(record.get("name") or "")
            if path.is_symlink() or not path.is_file():
                raise ValueError("seed feature cache array is missing")
            if path.stat().st_size != int(record.get("size", -1)):
                raise ValueError("seed feature cache array size mismatch")
            if _sha256_file(path) != str(record.get("sha256") or ""):
                raise ValueError("seed feature cache array hash mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = (int(self.pool.n_frames()), int(record["nfeats"]))
            if array.shape != expected_shape or array.dtype != np.dtype(np.float64):
                raise ValueError("seed feature cache array contract mismatch")
            if not np.all(np.isfinite(array)):
                raise ValueError("seed feature cache contains non-finite values")
            arrays[str(atom)] = array
        expected_atoms = set(str(atom) for atom in self.posterior._property_models)
        if set(arrays) != expected_atoms:
            raise ValueError("seed feature cache atom coverage mismatch")
        return arrays, payload

    def ensure_features(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if self._active_features is not None:
            return self._active_features
        identity = self._feature_identity()
        cache_id = _canonical_sha256(identity)
        features_root = self.root / "features"
        features_root.mkdir(parents=True, exist_ok=True)
        final = features_root / cache_id
        if final.exists() or final.is_symlink():
            try:
                arrays, manifest = self._read_feature_cache(final, identity)
                if self.progress is not None:
                    self.progress.cache("pool_features", "hit", cache_id=cache_id)
                self._active_features = (arrays, manifest)
                return self._active_features
            except Exception:
                _invalidate_directory(final)
                if self.progress is not None:
                    self.progress.cache("pool_features", "invalid_rebuild", cache_id=cache_id)

        building = features_root / (".building-" + cache_id)
        ledger_path = building / "BUILD.json"
        atom_records = {
            str(atom): {
                "name": "atom-" + str(position).zfill(4) + ".npy",
                "nfeats": int(model.nfeats),
            }
            for position, (atom, model) in enumerate(
                sorted(self.posterior._property_models.items())
            )
        }
        ledger: Dict[str, Any]
        arrays: Dict[str, np.ndarray] = {}
        if building.exists() or building.is_symlink():
            try:
                if building.is_symlink() or ledger_path.is_symlink():
                    raise ValueError("seed feature build path is symlinked")
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                if ledger.get("identity") != identity or ledger.get("atoms") != atom_records:
                    raise ValueError("seed feature build identity mismatch")
                if any(
                    (building / record["name"]).is_symlink()
                    for record in atom_records.values()
                ):
                    raise ValueError("seed feature build array is symlinked")
                arrays = {
                    atom: np.lib.format.open_memmap(
                        building / record["name"], mode="r+"
                    )
                    for atom, record in atom_records.items()
                }
                valid_chunks = []
                expected_start = 0
                total_rows = int(self.pool.n_frames())
                for chunk in list(ledger.get("chunks") or []):
                    start = int(chunk["start"])
                    stop = int(chunk["stop"])
                    if (
                        start != expected_start
                        or stop <= start
                        or stop > total_rows
                        or stop != min(total_rows, start + FEATURE_CHUNK_SIZE)
                    ):
                        break
                    if self._chunk_sha(arrays, start, stop) != str(chunk["sha256"]):
                        break
                    valid_chunks.append(dict(chunk))
                    expected_start = int(stop)
                ledger["chunks"] = valid_chunks
                ledger["completed_rows"] = (
                    int(valid_chunks[-1]["stop"]) if valid_chunks else 0
                )
                atomic_write_json(ledger_path, ledger)
                if self.progress is not None:
                    self.progress.cache(
                        "pool_features",
                        "resumed",
                        cache_id=cache_id,
                        completed=int(ledger["completed_rows"]),
                    )
            except Exception:
                _invalidate_directory(building)
                arrays = {}
        if not building.exists() and not building.is_symlink():
            building.mkdir(parents=True)
            for atom, record in atom_records.items():
                arrays[atom] = np.lib.format.open_memmap(
                    building / record["name"],
                    mode="w+",
                    dtype=np.float64,
                    shape=(int(self.pool.n_frames()), int(record["nfeats"])),
                )
            ledger = {
                "schema_version": 1,
                "identity": identity,
                "atoms": atom_records,
                "completed_rows": 0,
                "chunks": [],
            }
            atomic_write_json(ledger_path, ledger)
            if self.progress is not None:
                self.progress.cache("pool_features", "building", cache_id=cache_id)

        completed = int(ledger.get("completed_rows", 0))
        total = int(self.pool.n_frames())
        for start in range(completed, total, FEATURE_CHUNK_SIZE):
            stop = min(total, start + FEATURE_CHUNK_SIZE)
            for frame_id in range(start, stop):
                features = self.posterior.models.get_features_dict(
                    self.pool.frame(frame_id)
                )
                for atom, array in arrays.items():
                    values = np.asarray(features[atom], dtype=float).reshape(-1)
                    if values.shape != (array.shape[1],) or not np.all(
                        np.isfinite(values)
                    ):
                        raise ValueError(
                            "pool ALF feature contract failed for frame "
                            + str(frame_id)
                            + " atom "
                            + str(atom)
                        )
                    array[frame_id, :] = values
            for atom, array in arrays.items():
                array.flush()
                _fsync_file(building / atom_records[atom]["name"])
            chunk = {
                "start": int(start),
                "stop": int(stop),
                "sha256": self._chunk_sha(arrays, start, stop),
            }
            ledger["chunks"] = list(ledger.get("chunks") or []) + [chunk]
            ledger["completed_rows"] = int(stop)
            atomic_write_json(ledger_path, ledger)
            if self.progress is not None:
                self.progress.update(
                    "features",
                    completed=int(stop),
                    total=int(total),
                    cache_kind="pool_features",
                    cache_status="building",
                )

        files: Dict[str, Any] = {}
        for atom, record in atom_records.items():
            path = building / record["name"]
            files[atom] = {
                **record,
                "size": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        manifest = {
            "schema_version": SEED_FEATURE_CACHE_SCHEMA_VERSION,
            "cache_id": cache_id,
            "identity": identity,
            "files": files,
        }
        atomic_write_json(building / "SEED_FEATURE_CACHE.json", manifest)
        for array in arrays.values():
            array.flush()
            memory_map = getattr(array, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
        arrays.clear()
        ledger_path.unlink()
        if final.exists() or final.is_symlink():
            _invalidate_directory(final)
        os.replace(building, final)
        arrays, checked = self._read_feature_cache(final, identity)
        if self.progress is not None:
            self.progress.cache("pool_features", "published", cache_id=cache_id)
        self._active_features = (arrays, checked)
        return self._active_features

    def _variance_identity(
        self,
        *,
        feature_cache_id: str,
        eligible_indices: Sequence[int],
        chunk_size: int,
    ) -> Dict[str, Any]:
        eligible = np.asarray(eligible_indices, dtype=np.int64)
        return {
            "schema_version": SEED_VARIANCE_CACHE_SCHEMA_VERSION,
            "projection_version": SEED_POSTERIOR_PROJECTION_VERSION,
            "feature_cache_id": str(feature_cache_id),
            "model_set_sha256": self.model_set_sha256,
            "model_manifest_sha256": self.model_manifest_sha256,
            "property_name": str(self.posterior.property_name),
            "scaled": bool(self.posterior.scaled),
            "chunk_size": int(chunk_size),
            "eligible_count": int(eligible.size),
            "eligible_indices_sha256": hashlib.sha256(
                eligible.astype("<i8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "dtype": "float64",
        }

    def ensure_indexed_posterior(
        self,
        *,
        eligible_indices: Sequence[int],
        chunk_size: int,
        defer_variances: bool = False,
    ):
        if bool(defer_variances):
            return DeferredCachedPoolPosterior(
                self,
                eligible_indices=eligible_indices,
                chunk_size=int(chunk_size),
            )
        return self._ensure_indexed_posterior_now(
            eligible_indices=eligible_indices,
            chunk_size=int(chunk_size),
        )

    def _ensure_indexed_posterior_now(
        self,
        *,
        eligible_indices: Sequence[int],
        chunk_size: int,
    ) -> CachedPoolPosterior:
        feature_arrays, feature_manifest = self.ensure_features()
        eligible = np.asarray(eligible_indices, dtype=np.int64)
        if eligible.ndim != 1 or len(set(int(value) for value in eligible)) != int(
            eligible.size
        ):
            raise ValueError("seed variance eligibility indices are invalid")
        if np.any(eligible < 0) or np.any(eligible >= int(self.pool.n_frames())):
            raise ValueError("seed variance eligibility index is outside the pool")
        identity = self._variance_identity(
            feature_cache_id=str(feature_manifest["cache_id"]),
            eligible_indices=eligible,
            chunk_size=max(1, int(chunk_size)),
        )
        cache_id = _canonical_sha256(identity)
        variance_root = self.root / "variances"
        variance_root.mkdir(parents=True, exist_ok=True)
        data_path = variance_root / (cache_id + ".npy")
        manifest_path = variance_root / (cache_id + ".json")
        variances: Optional[np.ndarray] = None
        if manifest_path.is_file() and data_path.is_file():
            try:
                if manifest_path.is_symlink() or data_path.is_symlink():
                    raise ValueError("seed variance cache is symlinked")
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if payload.get("schema_version") != SEED_VARIANCE_CACHE_SCHEMA_VERSION:
                    raise ValueError("unsupported seed variance cache schema")
                if payload.get("identity") != identity:
                    raise ValueError("seed variance cache identity mismatch")
                if data_path.is_symlink() or data_path.stat().st_size != int(
                    payload.get("size", -1)
                ):
                    raise ValueError("seed variance cache size mismatch")
                if _sha256_file(data_path) != str(payload.get("sha256") or ""):
                    raise ValueError("seed variance cache hash mismatch")
                loaded = np.load(data_path, mmap_mode="r", allow_pickle=False)
                if loaded.shape != eligible.shape or loaded.dtype != np.dtype(np.float64):
                    raise ValueError("seed variance cache array contract mismatch")
                if not np.all(np.isfinite(loaded)) or np.any(loaded < 0.0):
                    raise ValueError("seed variance cache values are invalid")
                variances = loaded
                if self.progress is not None:
                    self.progress.cache("posterior_variances", "hit", cache_id=cache_id)
            except Exception:
                manifest_path.unlink(missing_ok=True)
                data_path.unlink(missing_ok=True)
                if self.progress is not None:
                    self.progress.cache(
                        "posterior_variances", "invalid_rebuild", cache_id=cache_id
                    )
        if variances is None:
            if self.progress is not None:
                self.progress.cache("posterior_variances", "building", cache_id=cache_id)
                self.progress.update(
                    "variance", force=True, completed=0, total=int(eligible.size)
                )
            size = max(1, int(chunk_size))
            building = variance_root / (".building-" + cache_id)
            build_path = building / "BUILD.json"
            build_data_path = building / "variances.npy"
            output = None
            ledger: Dict[str, Any]
            if building.exists() or building.is_symlink():
                try:
                    if building.is_symlink() or not building.is_dir():
                        raise ValueError("seed variance build path is invalid")
                    if build_path.is_symlink() or build_data_path.is_symlink():
                        raise ValueError("seed variance build file is symlinked")
                    ledger = json.loads(build_path.read_text(encoding="utf-8"))
                    if ledger.get("identity") != identity:
                        raise ValueError("seed variance build identity mismatch")
                    output = np.lib.format.open_memmap(build_data_path, mode="r+")
                    if (
                        output.shape != eligible.shape
                        or output.dtype != np.dtype(np.float64)
                    ):
                        raise ValueError("seed variance build array contract mismatch")
                    valid_chunks = []
                    expected_start = 0
                    for chunk in list(ledger.get("chunks") or []):
                        start = int(chunk["start"])
                        stop = int(chunk["stop"])
                        if (
                            start != expected_start
                            or stop != min(int(eligible.size), start + size)
                            or stop <= start
                        ):
                            break
                        observed_sha = hashlib.sha256(
                            np.ascontiguousarray(
                                output[start:stop], dtype="<f8"
                            ).tobytes(order="C")
                        ).hexdigest()
                        if observed_sha != str(chunk.get("sha256") or ""):
                            break
                        valid_chunks.append(dict(chunk))
                        expected_start = int(stop)
                    ledger["chunks"] = valid_chunks
                    ledger["completed_rows"] = int(expected_start)
                    atomic_write_json(build_path, ledger)
                    if self.progress is not None:
                        self.progress.cache(
                            "posterior_variances",
                            "resumed",
                            cache_id=cache_id,
                            completed=int(expected_start),
                        )
                except Exception:
                    if output is not None:
                        memory_map = getattr(output, "_mmap", None)
                        if memory_map is not None:
                            memory_map.close()
                    output = None
                    _invalidate_directory(building)
            if not building.exists() and not building.is_symlink():
                building.mkdir(parents=True)
                output = np.lib.format.open_memmap(
                    build_data_path,
                    mode="w+",
                    dtype=np.float64,
                    shape=eligible.shape,
                )
                ledger = {
                    "schema_version": 1,
                    "identity": identity,
                    "completed_rows": 0,
                    "chunks": [],
                }
                atomic_write_json(build_path, ledger)
            assert output is not None
            completed = int(ledger.get("completed_rows", 0))
            try:
                for start in range(completed, int(eligible.size), size):
                    stop = min(int(eligible.size), start + size)
                    selected_features = {
                        atom: np.asarray(values[eligible[start:stop]], dtype=float)
                        for atom, values in feature_arrays.items()
                    }
                    computed = self.posterior.variances_from_feature_arrays(
                        selected_features,
                        chunk_size=int(stop - start),
                    )
                    output[start:stop] = np.asarray(computed, dtype=np.float64)
                    output.flush()
                    _fsync_file(build_data_path)
                    chunk = {
                        "start": int(start),
                        "stop": int(stop),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(
                                output[start:stop], dtype="<f8"
                            ).tobytes(order="C")
                        ).hexdigest(),
                    }
                    ledger["chunks"] = list(ledger.get("chunks") or []) + [chunk]
                    ledger["completed_rows"] = int(stop)
                    atomic_write_json(build_path, ledger)
                    if self.progress is not None:
                        self.progress.update(
                            "variance",
                            completed=int(stop),
                            total=int(eligible.size),
                        )
            except Exception:
                output.flush()
                memory_map = getattr(output, "_mmap", None)
                if memory_map is not None:
                    memory_map.close()
                raise
            output.flush()
            memory_map = getattr(output, "_mmap", None)
            if memory_map is not None:
                memory_map.close()
            output = None
            os.replace(build_data_path, data_path)
            payload = {
                "schema_version": SEED_VARIANCE_CACHE_SCHEMA_VERSION,
                "cache_id": cache_id,
                "identity": identity,
                "size": int(data_path.stat().st_size),
                "sha256": _sha256_file(data_path),
            }
            atomic_write_json(manifest_path, payload)
            build_path.unlink(missing_ok=True)
            building.rmdir()
            variances = np.load(data_path, mmap_mode="r", allow_pickle=False)
            if self.progress is not None:
                self.progress.cache(
                    "posterior_variances", "published", cache_id=cache_id
                )
        return CachedPoolPosterior(
            posterior=self.posterior,
            feature_arrays=feature_arrays,
            variances=variances,
            eligible_indices=eligible,
            progress=self.progress,
            projection_directory=self.projection_directory,
            feature_cache_id=str(feature_manifest["cache_id"]),
            variance_cache_id=str(cache_id),
        )

    def finalise(self, *, selection_published: bool) -> None:
        """Remove iteration-local projections only after authoritative publish."""
        if not bool(selection_published):
            return
        workspace = self.projection_directory.parent
        if workspace.exists() or workspace.is_symlink():
            _invalidate_directory(workspace)

    def prune_superseded(
        self,
        *,
        active_feature_cache_id: str,
        active_variance_cache_id: str,
    ) -> None:
        """Bound derived namespaces after a successful seed selection."""
        features_root = self.root / "features"
        if features_root.is_dir() and not features_root.is_symlink():
            for child in features_root.iterdir():
                if child.name != str(active_feature_cache_id):
                    _invalidate_directory(child)
        variance_root = self.root / "variances"
        if variance_root.is_dir() and not variance_root.is_symlink():
            keep = {
                str(active_variance_cache_id) + ".npy",
                str(active_variance_cache_id) + ".json",
            }
            for child in variance_root.iterdir():
                if child.name not in keep:
                    _invalidate_directory(child)
        neighbours_root = self.root / "neighbours"
        if (
            self.active_neighbour_cache_id is not None
            and neighbours_root.is_dir()
            and not neighbours_root.is_symlink()
        ):
            keep_name = str(self.active_neighbour_cache_id) + ".json"
            for child in neighbours_root.iterdir():
                if child.name != keep_name:
                    _invalidate_directory(child)
        workspaces_root = self.root / "workspaces"
        if workspaces_root.is_dir() and not workspaces_root.is_symlink():
            for child in workspaces_root.iterdir():
                _invalidate_directory(child)
        factors_root = self.root / "model_factors"
        if factors_root.is_dir() and not factors_root.is_symlink():
            candidates = [
                child
                for child in factors_root.iterdir()
                if child.is_dir() and not child.is_symlink()
            ]
            predecessors = sorted(
                (
                    child
                    for child in candidates
                    if child.name != self.model_set_sha256[:24]
                ),
                key=lambda child: child.stat().st_mtime_ns,
                reverse=True,
            )
            keep = {self.model_set_sha256[:24]}
            if predecessors:
                keep.add(predecessors[0].name)
            for child in factors_root.iterdir():
                if child.name not in keep:
                    _invalidate_directory(child)


def restore_ariadne_model_factors(
    campaign_dir: Path,
    *,
    pool: Any,
    models: Any,
    model_set: Any,
    iteration: int,
) -> Dict[str, str]:
    """Restore current factors before constructing an ARIADNE posterior."""
    from types import SimpleNamespace

    property_models = {
        str(model.atom): model
        for model in models
        if str(model.type) == "iqa"
    }
    if not property_models:
        raise ValueError("ARIADNE requires current IQA models")
    file_hashes = {
        str(task.atom): str(task.model.sha256)
        for task in model_set.tasks
        if str(task.property) == "iqa"
    }
    if set(property_models) != set(file_hashes):
        raise ValueError("ARIADNE model-factor inventory is incomplete")
    cache = SeedSelectionRuntimeCache(
        Path(campaign_dir),
        pool=pool,
        posterior=SimpleNamespace(_property_models=property_models),
        model_set_sha256=str(model_set.model_set_sha256),
        model_manifest_sha256=str(model_set.head_manifest_sha256),
        iteration=int(iteration),
        model_file_sha256_by_atom=file_hashes,
    )
    return cache.ensure_model_factors()


__all__ = [
    "CachedPoolPosterior",
    "DeferredCachedPoolPosterior",
    "SEED_MODEL_FACTOR_CACHE_SCHEMA_VERSION",
    "SEED_SELECTION_PROGRESS_STAGES",
    "SeedSelectionProgressReporter",
    "SeedSelectionRuntimeCache",
    "restore_ariadne_model_factors",
    "finalise_seed_selection_workspace",
]
