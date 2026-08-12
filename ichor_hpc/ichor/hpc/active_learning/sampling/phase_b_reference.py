"""Authority-bound derived coordinates for Phase B novelty filtering."""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import Atom, Atoms
from ichor.core.common.units import AtomicDistance
from ichor.core.files import PointDirectory

from ..daemon.filesystem import campaign_owned_path
from ..daemon.state import (
    _fsync_file_descriptor,
    _fsync_parent_dir,
    atomic_write_json,
)
from ..strict_json import strict_json as json
from ..versioning.manifest import sha256_file
from ..versioning.reference_data import ReferenceDataVersioning


PHASE_B_REFERENCE_CACHE_SCHEMA_VERSION = 1
PHASE_B_REFERENCE_PARSER_IDENTITY = "point_directory_xyz_then_wfn_angstrom_v1"
PHASE_B_REFERENCE_CACHE_ROOT = (
    Path(".DATA") / "CACHE" / "PHASE_B_REFERENCE_COORDINATES"
)
PHASE_B_REFERENCE_CACHE_DATA = "coordinates.npy"
PHASE_B_REFERENCE_CACHE_MANIFEST = "CACHE_MANIFEST.json"


class PhaseBReferenceCacheError(ValueError):
    """Raised when reference-coordinate cache evidence is unsafe."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_stat(path: Path) -> Tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_stat(path: Path) -> Tuple[int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


@dataclass(frozen=True)
class PhaseBReferenceSourceAnchor:
    global_ordinal: int
    pointdir: Path
    pointdir_stat: Tuple[int, int, int, int]
    source: Path
    source_kind: str
    source_stat: Tuple[int, int, int, int, int]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "global_ordinal": int(self.global_ordinal),
            "pointdir": str(self.pointdir),
            "pointdir_stat": list(self.pointdir_stat),
            "source": str(self.source),
            "source_kind": str(self.source_kind),
            "source_stat": list(self.source_stat),
        }


@dataclass(frozen=True)
class PhaseBReferenceCoordinates:
    campaign_dir: Path
    reference_version: int
    campaign_uid: str
    cumulative_view_sha256: str
    head_manifest_sha256: str
    coordinates: np.ndarray
    atom_types: Tuple[str, ...]
    atom_indices: Tuple[int, ...]
    atom_units: Tuple[str, ...]
    masses: Tuple[float, ...]
    anchors: Tuple[PhaseBReferenceSourceAnchor, ...]
    cache_status: str
    cache_directory: Optional[Path]

    @property
    def n_references(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.coordinates.shape[1]) if self.coordinates.ndim == 3 else 0

    def frame(self, index: int) -> Atoms:
        row = np.asarray(self.coordinates[int(index)], dtype=np.float64)
        return Atoms(
            [
                Atom(
                    atom_type,
                    float(coordinate[0]),
                    float(coordinate[1]),
                    float(coordinate[2]),
                    index=int(atom_index),
                    units=AtomicDistance(unit),
                )
                for atom_type, atom_index, unit, coordinate in zip(
                    self.atom_types,
                    self.atom_indices,
                    self.atom_units,
                    row,
                )
            ]
        )

    def to_atoms_list(self) -> list[Atoms]:
        return [self.frame(index) for index in range(self.n_references)]

    def assert_sources_unchanged(self, *, workers: int = 8) -> None:
        failures: Dict[int, str] = {}

        def check(anchor: PhaseBReferenceSourceAnchor) -> Optional[str]:
            try:
                point_lstat = anchor.pointdir.lstat()
                if anchor.pointdir.is_symlink() or not stat.S_ISDIR(
                    point_lstat.st_mode
                ):
                    return "pointdir is missing or unsafe"
                if _directory_stat(anchor.pointdir) != anchor.pointdir_stat:
                    return "pointdir identity changed"
                source_lstat = anchor.source.lstat()
                if anchor.source.is_symlink() or not stat.S_ISREG(
                    source_lstat.st_mode
                ):
                    return "coordinate source is missing or unsafe"
                if _file_stat(anchor.source) != anchor.source_stat:
                    return "coordinate source identity changed"
                return None
            except OSError as exc:
                return type(exc).__name__ + ": " + str(exc)

        with ThreadPoolExecutor(
            max_workers=max(1, min(int(workers), len(self.anchors) or 1))
        ) as pool:
            futures = {pool.submit(check, anchor): anchor for anchor in self.anchors}
            for future in as_completed(futures):
                anchor = futures[future]
                message = future.result()
                if message is not None:
                    failures[int(anchor.global_ordinal)] = message
        if failures:
            ordinal = sorted(failures)[0]
            raise PhaseBReferenceCacheError(
                "Phase B reference source changed at global ordinal "
                + str(ordinal)
                + ": "
                + failures[ordinal]
            )


@dataclass(frozen=True)
class PhaseBFileAnchor:
    path: Path
    size: int
    sha256: str
    stat_identity: Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class PhaseBDiversityAuthorityContext:
    campaign_dir: Path
    iteration: int
    campaign_uid: str
    reference_view: Any
    reference_coordinates: PhaseBReferenceCoordinates
    ariadne_manifest: Mapping[str, Any]
    candidate_frames: Tuple[Atoms, ...]
    candidate_records: Tuple[Mapping[str, Any], ...]
    candidate_order_sha256: str
    handoff_anchors: Tuple[PhaseBFileAnchor, ...]
    artifact_snapshot: Any

    def assert_unchanged(self) -> None:
        self.artifact_snapshot.assert_reference_head_unchanged(
            self.campaign_dir,
            reference_version=int(self.reference_view.version),
            purpose="Phase B submission",
        )
        versioning = ReferenceDataVersioning(
            self.campaign_dir / "QM_REFERENCE_DATA"
        )
        if versioning.current_version() != int(self.reference_view.version):
            raise PhaseBReferenceCacheError(
                "Phase B reference-data pointer changed during submission"
            )
        self.reference_coordinates.assert_sources_unchanged()
        failures = []
        for anchor in self.handoff_anchors:
            try:
                if anchor.path.is_symlink() or not anchor.path.is_file():
                    failures.append((str(anchor.path), "missing or unsafe"))
                    continue
                if _file_stat(anchor.path) != anchor.stat_identity:
                    failures.append((str(anchor.path), "file identity changed"))
                    continue
                if sha256_file(anchor.path) != anchor.sha256:
                    failures.append((str(anchor.path), "content changed"))
            except OSError as exc:
                failures.append(
                    (str(anchor.path), type(exc).__name__ + ": " + str(exc))
                )
        if failures:
            path, detail = sorted(failures)[0]
            raise PhaseBReferenceCacheError(
                "Phase B ARIADNE handoff changed: " + path + ": " + detail
            )


def _cache_identity(view: Any) -> Dict[str, Any]:
    return {
        "campaign_uid": str(view.campaign_uid),
        "reference_version": int(view.version),
        "cumulative_view_sha256": str(view.cumulative_view_sha256),
        "head_manifest_sha256": str(view.head_manifest_sha256),
        "ordered_entries_sha256": _canonical_sha256(
            [entry.identity_payload() for entry in view.entries]
        ),
        "n_references": int(len(view.entries)),
        "parser_identity": PHASE_B_REFERENCE_PARSER_IDENTITY,
        "dtype": np.dtype(np.float64).str,
        "byteorder": sys.byteorder,
        "platform": platform.system(),
        "machine": platform.machine(),
        "numpy_version": str(np.__version__),
    }


def _cache_paths(campaign: Path, view: Any) -> Dict[str, Path]:
    root = campaign_owned_path(campaign, PHASE_B_REFERENCE_CACHE_ROOT)
    digest = str(view.cumulative_view_sha256)
    directory = root / digest
    return {
        "root": root,
        "directory": directory,
        "data": directory / PHASE_B_REFERENCE_CACHE_DATA,
        "manifest": directory / PHASE_B_REFERENCE_CACHE_MANIFEST,
        "lock": root / (digest + ".lock"),
    }


def _source_anchor_from_payload(payload: Mapping[str, Any]) -> PhaseBReferenceSourceAnchor:
    try:
        return PhaseBReferenceSourceAnchor(
            global_ordinal=int(payload["global_ordinal"]),
            pointdir=Path(str(payload["pointdir"])),
            pointdir_stat=tuple(int(value) for value in payload["pointdir_stat"]),
            source=Path(str(payload["source"])),
            source_kind=str(payload["source_kind"]),
            source_stat=tuple(int(value) for value in payload["source_stat"]),
        )
    except Exception as exc:
        raise PhaseBReferenceCacheError(
            "Phase B reference cache source anchor is malformed"
        ) from exc


def _read_cache(campaign: Path, view: Any) -> PhaseBReferenceCoordinates:
    paths = _cache_paths(campaign, view)
    directory = paths["directory"]
    if directory.is_symlink() or not directory.is_dir():
        raise PhaseBReferenceCacheError("Phase B reference cache directory is unsafe")
    manifest_path = paths["manifest"]
    data_path = paths["data"]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PhaseBReferenceCacheError("Phase B reference cache manifest is missing")
    if data_path.is_symlink() or not data_path.is_file():
        raise PhaseBReferenceCacheError("Phase B reference cache data is missing")
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"), source=manifest_path
    )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "identity",
        "atom_identity",
        "sources",
        "data",
    }:
        raise PhaseBReferenceCacheError("Phase B reference cache fields are invalid")
    if payload.get("schema_version") != PHASE_B_REFERENCE_CACHE_SCHEMA_VERSION:
        raise PhaseBReferenceCacheError("Phase B reference cache schema is unsupported")
    if payload.get("identity") != _cache_identity(view):
        raise PhaseBReferenceCacheError("Phase B reference cache identity mismatch")
    atom_identity = payload.get("atom_identity")
    sources = payload.get("sources")
    data = payload.get("data")
    if not isinstance(atom_identity, dict) or not isinstance(sources, list):
        raise PhaseBReferenceCacheError("Phase B reference atom evidence is invalid")
    if not isinstance(data, dict) or data.get("name") != PHASE_B_REFERENCE_CACHE_DATA:
        raise PhaseBReferenceCacheError("Phase B reference cache data record is invalid")
    anchors = tuple(_source_anchor_from_payload(item) for item in sources)
    if [anchor.global_ordinal for anchor in anchors] != list(range(len(view.entries))):
        raise PhaseBReferenceCacheError("Phase B reference cache order is invalid")
    if any(
        len(anchor.pointdir_stat) != 4
        or len(anchor.source_stat) != 5
        or anchor.source_kind not in {"xyz", "wfn"}
        for anchor in anchors
    ):
        raise PhaseBReferenceCacheError(
            "Phase B reference cache source evidence is invalid"
        )
    normalised_anchors = []
    for entry, anchor in zip(view.entries, anchors):
        expected_pointdir = campaign_owned_path(
            campaign,
            Path(entry.pointdir_path),
        ).resolve()
        try:
            observed_pointdir = campaign_owned_path(
                campaign,
                anchor.pointdir,
            ).resolve()
            observed_source = campaign_owned_path(
                campaign,
                anchor.source,
            ).resolve()
            observed_source.relative_to(observed_pointdir)
        except (OSError, ValueError) as exc:
            raise PhaseBReferenceCacheError(
                "Phase B reference cache source path is unsafe"
            ) from exc
        if observed_pointdir != expected_pointdir:
            raise PhaseBReferenceCacheError(
                "Phase B reference cache pointdir identity mismatch"
            )
        normalised_anchors.append(
            PhaseBReferenceSourceAnchor(
                global_ordinal=int(anchor.global_ordinal),
                pointdir=observed_pointdir,
                pointdir_stat=tuple(anchor.pointdir_stat),
                source=observed_source,
                source_kind=str(anchor.source_kind),
                source_stat=tuple(anchor.source_stat),
            )
        )
    anchors = tuple(normalised_anchors)
    expected_shape = (len(view.entries), len(atom_identity.get("types") or []), 3)
    if data.get("shape") != list(expected_shape):
        raise PhaseBReferenceCacheError("Phase B reference cache shape is invalid")
    if data.get("dtype") != np.dtype(np.float64).str:
        raise PhaseBReferenceCacheError("Phase B reference cache dtype is invalid")
    if int(data.get("size", -1)) != int(data_path.stat().st_size):
        raise PhaseBReferenceCacheError("Phase B reference cache size mismatch")
    if str(data.get("sha256") or "") != sha256_file(data_path):
        raise PhaseBReferenceCacheError("Phase B reference cache hash mismatch")
    coordinates = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if coordinates.shape != expected_shape or coordinates.dtype != np.dtype(np.float64):
        raise PhaseBReferenceCacheError("Phase B reference array contract mismatch")
    if not np.all(np.isfinite(coordinates)):
        raise PhaseBReferenceCacheError("Phase B reference coordinates are non-finite")
    result = PhaseBReferenceCoordinates(
        campaign_dir=campaign,
        reference_version=int(view.version),
        campaign_uid=str(view.campaign_uid),
        cumulative_view_sha256=str(view.cumulative_view_sha256),
        head_manifest_sha256=str(view.head_manifest_sha256),
        coordinates=coordinates,
        atom_types=tuple(str(value) for value in atom_identity.get("types") or []),
        atom_indices=tuple(int(value) for value in atom_identity.get("indices") or []),
        atom_units=tuple(str(value) for value in atom_identity.get("units") or []),
        masses=tuple(float(value) for value in atom_identity.get("masses") or []),
        anchors=anchors,
        cache_status="hit",
        cache_directory=directory,
    )
    if not (
        len(result.atom_types)
        == len(result.atom_indices)
        == len(result.atom_units)
        == len(result.masses)
        == result.n_atoms
    ):
        raise PhaseBReferenceCacheError("Phase B reference atom identity is incomplete")
    if any(unit != AtomicDistance.Angstroms.value for unit in result.atom_units):
        raise PhaseBReferenceCacheError("Phase B reference cache units are invalid")
    result.assert_sources_unchanged()
    return result


def _parse_reference_entry(entry: Any) -> Tuple[np.ndarray, Tuple[Any, ...], PhaseBReferenceSourceAnchor]:
    pointdir = Path(entry.pointdir_path)
    point_lstat = pointdir.lstat()
    if pointdir.is_symlink() or not stat.S_ISDIR(point_lstat.st_mode):
        raise PhaseBReferenceCacheError(
            "committed Phase B reference pointdir is missing or unsafe: "
            + str(pointdir)
        )
    before_directory = _directory_stat(pointdir)
    try:
        wrapped = PointDirectory(pointdir)
        atoms = wrapped.atoms
        source_object = wrapped.xyz if wrapped.xyz else wrapped.wfn
        source_kind = "xyz" if wrapped.xyz else "wfn"
        source = Path(source_object.path)
    except Exception as exc:
        raise PhaseBReferenceCacheError(
            "committed QM reference point cannot be parsed: " + str(pointdir)
        ) from exc
    try:
        source.resolve().relative_to(pointdir.resolve())
    except ValueError as exc:
        raise PhaseBReferenceCacheError(
            "Phase B reference coordinate source escapes its pointdir"
        ) from exc
    if source.is_symlink() or not source.is_file():
        raise PhaseBReferenceCacheError(
            "Phase B reference coordinate source is missing or unsafe"
        )
    before_source = _file_stat(source)
    coordinates = np.asarray(atoms.coordinates, dtype=np.float64)
    identity = (
        tuple(str(atom.type) for atom in atoms),
        tuple(int(atom.index) for atom in atoms),
        tuple(str(atom.units.value) for atom in atoms),
        tuple(float(atom.mass) for atom in atoms),
    )
    if coordinates.shape != (len(atoms), 3) or not np.all(np.isfinite(coordinates)):
        raise PhaseBReferenceCacheError("Phase B reference coordinates are invalid")
    if any(unit != AtomicDistance.Angstroms.value for unit in identity[2]):
        raise PhaseBReferenceCacheError("Phase B reference coordinates are not Angstroms")
    if _directory_stat(pointdir) != before_directory or _file_stat(source) != before_source:
        raise PhaseBReferenceCacheError(
            "Phase B reference source changed while it was parsed"
        )
    return (
        coordinates,
        identity,
        PhaseBReferenceSourceAnchor(
            global_ordinal=int(entry.global_ordinal),
            pointdir=pointdir.resolve(),
            pointdir_stat=before_directory,
            source=source.resolve(),
            source_kind=source_kind,
            source_stat=before_source,
        ),
    )


def _build_coordinates(
    campaign: Path,
    view: Any,
    *,
    workers: int,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]],
) -> PhaseBReferenceCoordinates:
    total = len(view.entries)
    if total == 0:
        coordinates = np.empty((0, 0, 3), dtype=np.float64)
        return PhaseBReferenceCoordinates(
            campaign_dir=campaign,
            reference_version=int(view.version),
            campaign_uid=str(view.campaign_uid),
            cumulative_view_sha256=str(view.cumulative_view_sha256),
            head_manifest_sha256=str(view.head_manifest_sha256),
            coordinates=coordinates,
            atom_types=(),
            atom_indices=(),
            atom_units=(),
            masses=(),
            anchors=(),
            cache_status="built_in_memory",
            cache_directory=None,
        )
    parsed: Dict[int, Tuple[np.ndarray, Tuple[Any, ...], PhaseBReferenceSourceAnchor]] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), total))) as pool:
        futures = {
            pool.submit(_parse_reference_entry, entry): int(entry.global_ordinal)
            for entry in view.entries
        }
        completed = 0
        for future in as_completed(futures):
            ordinal = futures[future]
            parsed[ordinal] = future.result()
            completed += 1
            if progress_callback is not None and (completed == total or completed % 64 == 0):
                try:
                    progress_callback(
                        "phase_b_reference_coordinates",
                        {
                            "completed": completed,
                            "total": total,
                            "unit": "reference points",
                            "cache_status": "building",
                        },
                    )
                except Exception:
                    pass
    if sorted(parsed) != list(range(total)):
        raise PhaseBReferenceCacheError("Phase B reference ordinals are not contiguous")
    first_identity = parsed[0][1]
    for ordinal in range(total):
        if parsed[ordinal][1] != first_identity:
            raise PhaseBReferenceCacheError(
                "Phase B reference atom identity differs at ordinal " + str(ordinal)
            )
    coordinates = np.stack(
        [parsed[ordinal][0] for ordinal in range(total)], axis=0
    ).astype(np.float64, copy=False)
    coordinates.setflags(write=False)
    return PhaseBReferenceCoordinates(
        campaign_dir=campaign,
        reference_version=int(view.version),
        campaign_uid=str(view.campaign_uid),
        cumulative_view_sha256=str(view.cumulative_view_sha256),
        head_manifest_sha256=str(view.head_manifest_sha256),
        coordinates=coordinates,
        atom_types=tuple(first_identity[0]),
        atom_indices=tuple(first_identity[1]),
        atom_units=tuple(first_identity[2]),
        masses=tuple(first_identity[3]),
        anchors=tuple(parsed[index][2] for index in range(total)),
        cache_status="built_in_memory",
        cache_directory=None,
    )


def _publish_cache(
    campaign: Path,
    view: Any,
    data: PhaseBReferenceCoordinates,
) -> PhaseBReferenceCoordinates:
    paths = _cache_paths(campaign, view)
    paths["root"].mkdir(parents=True, exist_ok=True)
    building = paths["root"] / (".b-" + uuid.uuid4().hex[:12])
    building.mkdir()
    try:
        data_path = building / PHASE_B_REFERENCE_CACHE_DATA
        with data_path.open("xb") as handle:
            np.save(
                handle,
                np.asarray(data.coordinates, dtype=np.float64),
                allow_pickle=False,
            )
            handle.flush()
            _fsync_file_descriptor(handle.fileno())
        payload = {
            "schema_version": PHASE_B_REFERENCE_CACHE_SCHEMA_VERSION,
            "identity": _cache_identity(view),
            "atom_identity": {
                "types": list(data.atom_types),
                "indices": [int(value) for value in data.atom_indices],
                "units": list(data.atom_units),
                "masses": [float(value) for value in data.masses],
            },
            "sources": [anchor.to_payload() for anchor in data.anchors],
            "data": {
                "name": PHASE_B_REFERENCE_CACHE_DATA,
                "size": int(data_path.stat().st_size),
                "sha256": sha256_file(data_path),
                "shape": list(data.coordinates.shape),
                "dtype": np.dtype(np.float64).str,
            },
        }
        atomic_write_json(building / PHASE_B_REFERENCE_CACHE_MANIFEST, payload)
        if paths["directory"].exists() or paths["directory"].is_symlink():
            try:
                return _read_cache(campaign, view)
            except (OSError, ValueError):
                if paths["directory"].is_symlink() or not paths["directory"].is_dir():
                    raise PhaseBReferenceCacheError(
                        "invalid Phase B cache namespace is unsafe to replace"
                    )
                retired = paths["root"] / (".bad-" + uuid.uuid4().hex[:12])
                os.replace(paths["directory"], retired)
                _fsync_parent_dir(retired)
        os.replace(building, paths["directory"])
        _fsync_parent_dir(paths["directory"])
        return _read_cache(campaign, view)
    finally:
        if building.exists() and not building.is_symlink():
            shutil.rmtree(building, ignore_errors=True)


def load_phase_b_reference_coordinates(
    campaign_dir: Path,
    *,
    reference_view: Any = None,
    workers: int = 8,
    progress_callback: Optional[Callable[[str, Mapping[str, Any]], None]] = None,
) -> PhaseBReferenceCoordinates:
    """Load or lazily build exact committed coordinates for Phase B."""
    campaign = Path(campaign_dir).resolve()
    view = reference_view
    if view is None:
        versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
        current = versioning.current_version()
        if current is None:
            return PhaseBReferenceCoordinates(
                campaign_dir=campaign,
                reference_version=-1,
                campaign_uid="",
                cumulative_view_sha256="",
                head_manifest_sha256="",
                coordinates=np.empty((0, 0, 3), dtype=np.float64),
                atom_types=(),
                atom_indices=(),
                atom_units=(),
                masses=(),
                anchors=(),
                cache_status="empty",
                cache_directory=None,
            )
        view = versioning.resolve(int(current), verification="index")
    paths = _cache_paths(campaign, view)
    try:
        return _read_cache(campaign, view)
    except (OSError, ValueError):
        pass
    built = _build_coordinates(
        campaign,
        view,
        workers=workers,
        progress_callback=progress_callback,
    )
    try:
        import portalocker

        paths["root"].mkdir(parents=True, exist_ok=True)
        if paths["lock"].is_symlink():
            raise PhaseBReferenceCacheError("Phase B reference cache lock is unsafe")
        with portalocker.Lock(str(paths["lock"]), mode="a", timeout=600):
            try:
                return _read_cache(campaign, view)
            except (OSError, ValueError):
                return _publish_cache(campaign, view, built)
    except Exception:
        return built


def build_phase_b_diversity_authority_context(
    campaign_dir: Path,
    iteration: int,
    *,
    expected_campaign_uid: str,
    expected_reference_version: int,
    artifact_snapshot: Any,
    progress_callback: Optional[Callable[..., None]] = None,
) -> PhaseBDiversityAuthorityContext:
    """Authenticate Phase B handoff and reference authority once."""
    from ..handoff_manifests import (
        ariadne_batch_decision_path,
        ariadne_results_path,
        authoritative_ariadne_candidate_frames,
    )
    from ..layout import active_iteration_dir

    campaign = Path(campaign_dir).resolve()
    view = artifact_snapshot.reference_view(int(expected_reference_version))
    if str(view.campaign_uid) != str(expected_campaign_uid):
        raise PhaseBReferenceCacheError("Phase B reference campaign identity mismatch")
    versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
    if versioning.current_version() != int(expected_reference_version):
        raise PhaseBReferenceCacheError("Phase B reference pointer is stale")

    def report(stage: str, fields: Mapping[str, Any]) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(str(stage), **dict(fields))
        except TypeError:
            try:
                progress_callback(str(stage), dict(fields))
            except Exception:
                pass
        except Exception:
            pass

    reference_data = load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        progress_callback=report,
    )
    iter_dir = active_iteration_dir(campaign, int(iteration))
    manifest, frames, records = authoritative_ariadne_candidate_frames(
        campaign,
        int(iteration),
    )
    order_material = [
        {
            "seed_id": int(record["seed_id"]),
            "seed_uid": str(record["seed_uid"]),
            "result_sha256": str(record.get("result_sha256") or ""),
        }
        for record in records
    ]
    anchor_paths = [
        ariadne_results_path(iter_dir),
        ariadne_batch_decision_path(iter_dir),
        *(Path(str(record["result_json"])) for record in records),
    ]
    handoff_anchors = []
    seen = set()
    for path in anchor_paths:
        owned = campaign_owned_path(campaign, path)
        resolved = owned.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if owned.is_symlink() or not owned.is_file():
            raise PhaseBReferenceCacheError(
                "Phase B ARIADNE handoff file is missing or unsafe: " + str(owned)
            )
        handoff_anchors.append(
            PhaseBFileAnchor(
                path=resolved,
                size=int(owned.stat().st_size),
                sha256=sha256_file(owned),
                stat_identity=_file_stat(owned),
            )
        )
    context = PhaseBDiversityAuthorityContext(
        campaign_dir=campaign,
        iteration=int(iteration),
        campaign_uid=str(expected_campaign_uid),
        reference_view=view,
        reference_coordinates=reference_data,
        ariadne_manifest=manifest,
        candidate_frames=tuple(frame.copy() for frame in frames),
        candidate_records=tuple(dict(record) for record in records),
        candidate_order_sha256=_canonical_sha256(order_material),
        handoff_anchors=tuple(handoff_anchors),
        artifact_snapshot=artifact_snapshot,
    )
    context.assert_unchanged()
    return context


def prewarm_phase_b_reference_cache(
    campaign_dir: Path,
    reference_view: Any,
) -> Optional[PhaseBReferenceCoordinates]:
    """Best-effort incremental cache publication after a reference commit."""
    try:
        campaign = Path(campaign_dir).resolve()
        try:
            return _read_cache(campaign, reference_view)
        except (OSError, ValueError):
            pass
        if int(reference_view.version) <= 0:
            return load_phase_b_reference_coordinates(
                campaign,
                reference_view=reference_view,
            )
        versioning = ReferenceDataVersioning(campaign / "QM_REFERENCE_DATA")
        parent_view = versioning.resolve(
            int(reference_view.version) - 1,
            verification="index",
        )
        parent = _read_cache(campaign, parent_view)
        prefix = tuple(
            entry.identity_payload()
            for entry in reference_view.entries[: len(parent_view.entries)]
        )
        if prefix != tuple(entry.identity_payload() for entry in parent_view.entries):
            raise PhaseBReferenceCacheError(
                "Phase B reference cache parent is not an exact prefix"
            )
        new_entries = tuple(reference_view.entries[len(parent_view.entries) :])
        if not new_entries:
            raise PhaseBReferenceCacheError(
                "Phase B reference version contains no new entries"
            )
        parsed: Dict[
            int,
            Tuple[np.ndarray, Tuple[Any, ...], PhaseBReferenceSourceAnchor],
        ] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(8, len(new_entries)))) as pool:
            futures = {
                pool.submit(_parse_reference_entry, entry): int(entry.global_ordinal)
                for entry in new_entries
            }
            for future in as_completed(futures):
                ordinal = futures[future]
                parsed[ordinal] = future.result()
        expected_ordinals = list(range(len(parent_view.entries), len(reference_view.entries)))
        if sorted(parsed) != expected_ordinals:
            raise PhaseBReferenceCacheError(
                "Phase B incremental reference ordinals are invalid"
            )
        parent_identity = (
            parent.atom_types,
            parent.atom_indices,
            parent.atom_units,
            parent.masses,
        )
        for ordinal in expected_ordinals:
            if parsed[ordinal][1] != parent_identity:
                raise PhaseBReferenceCacheError(
                    "Phase B incremental reference atom identity mismatch"
                )
        coordinates = np.concatenate(
            [
                np.asarray(parent.coordinates, dtype=np.float64),
                np.stack([parsed[index][0] for index in expected_ordinals], axis=0),
            ],
            axis=0,
        )
        coordinates.setflags(write=False)
        combined = PhaseBReferenceCoordinates(
            campaign_dir=campaign,
            reference_version=int(reference_view.version),
            campaign_uid=str(reference_view.campaign_uid),
            cumulative_view_sha256=str(reference_view.cumulative_view_sha256),
            head_manifest_sha256=str(reference_view.head_manifest_sha256),
            coordinates=coordinates,
            atom_types=parent.atom_types,
            atom_indices=parent.atom_indices,
            atom_units=parent.atom_units,
            masses=parent.masses,
            anchors=parent.anchors
            + tuple(parsed[index][2] for index in expected_ordinals),
            cache_status="built_incrementally",
            cache_directory=None,
        )
        paths = _cache_paths(campaign, reference_view)
        paths["root"].mkdir(parents=True, exist_ok=True)
        if paths["lock"].is_symlink():
            raise PhaseBReferenceCacheError("Phase B reference cache lock is unsafe")
        import portalocker

        with portalocker.Lock(str(paths["lock"]), mode="a", timeout=600):
            try:
                return _read_cache(campaign, reference_view)
            except (OSError, ValueError):
                return _publish_cache(campaign, reference_view, combined)
    except Exception:
        try:
            return load_phase_b_reference_coordinates(
                Path(campaign_dir),
                reference_view=reference_view,
            )
        except Exception:
            return None


__all__ = [
    "PHASE_B_REFERENCE_CACHE_SCHEMA_VERSION",
    "PhaseBDiversityAuthorityContext",
    "PhaseBReferenceCacheError",
    "PhaseBReferenceCoordinates",
    "build_phase_b_diversity_authority_context",
    "load_phase_b_reference_coordinates",
    "prewarm_phase_b_reference_cache",
]
