"""Trajectory pool: the immutable, content-addressed/hashed input to every campaign.

The MD / metadynamics trajectory is the campaign most precious input.
This module wraps it in an artefact with stable frame IDs, a SHA-256
content hash, and an atomic manifest that every iteration manifest can
transitively prove descendance from.

Layout on disk under <campaign_dir>:

    pool.xyz                    # canonical user trajectory
    .DATA/TRAJECTORY/
        pool.manifest.json      # SHA + n_frames + atom_types + masses + imported_iso

Frame IDs are the stable load-order positions in the canonical pool.xyz,
i.e. frame_id in range(0, n_frames). Once the pool is imported the IDs
are fixed for the lifetime of the campaign. Re-importing into a populated
campaign dir refuses (raises FileExistsError) -- a different MD pool means
a new campaign.

The public surface is intentionally small:

    TrajectoryPool.import_from(source, campaign_dir)  # copy + manifest
    TrajectoryPool.load(campaign_dir)                 # read existing
    pool.n_frames(), pool.frame(frame_id), pool.frame_ids(), pool.to_atoms_list()

Both select_local_neighbours and the seed-selection wiring
consume the pool via these methods; the daemon never touches the canonical
file directly.
"""
from __future__ import annotations

from ..strict_json import strict_json as json
import os
import platform
import shutil
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ichor.core.atoms import Atom, Atoms
from ichor.core.files.xyz.strict_xyz import iter_xyz_frames

from ..daemon.filesystem import campaign_owned_path
from ..daemon.state import (
    _fsync_file_descriptor,
    _fsync_parent_dir,
    atomic_write_json,
)
from ..versioning.manifest import sha256_file


__all__ = [
    "TrajectoryPool",
    "TrajectoryPoolManifest",
    "POOL_SUBDIR",
    "POOL_XYZ_FILENAME",
    "POOL_MANIFEST_FILENAME",
    "POOL_SCHEMA_VERSION",
    "POOL_COORDINATE_CACHE_SUBDIR",
    "POOL_COORDINATE_CACHE_FILENAME",
    "POOL_COORDINATE_CACHE_MANIFEST_FILENAME",
]


POOL_SUBDIR = Path(".DATA") / "TRAJECTORY"
POOL_XYZ_FILENAME = "pool.xyz"
POOL_MANIFEST_FILENAME = "pool.manifest.json"
POOL_IMPORT_TRANSACTION_FILENAME = "pool.import.transaction.json"
POOL_SCHEMA_VERSION = 2
POOL_IMPORT_TRANSACTION_SCHEMA_VERSION = 1
POOL_COORDINATE_CACHE_SCHEMA_VERSION = 1
POOL_COORDINATE_ENCODING_VERSION = 1
POOL_COORDINATE_PARSER_CONTRACT = "strict_xyz_stream_v1"
POOL_COORDINATE_CACHE_SUBDIR = Path(".DATA") / "CACHE" / "TRAJECTORY_POOL"
POOL_COORDINATE_CACHE_FILENAME = "coordinates.npy"
POOL_COORDINATE_CACHE_MANIFEST_FILENAME = "CACHE_MANIFEST.json"


def _validated_frames(path: Path, *, source_label: Path) -> tuple:
    frames = [atoms.copy() for atoms in iter_xyz_frames(path)]
    if not frames:
        raise ValueError("trajectory has zero frames: " + str(source_label))
    head = frames[0]
    atom_types = tuple(atom.type for atom in head)
    masses = tuple(float(atom.mass) for atom in head)
    for index, atoms in enumerate(frames):
        observed = tuple(atom.type for atom in atoms)
        if observed != atom_types:
            raise ValueError(
                "frame "
                + str(index)
                + " atom types "
                + repr(observed)
                + " disagree with frame 0 "
                + repr(atom_types)
            )
        coordinates = np.asarray(atoms.coordinates, dtype=float)
        if coordinates.shape != (len(head), 3):
            raise ValueError("frame " + str(index) + " coordinate shape is invalid")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("frame " + str(index) + " contains non-finite coordinates")
    return frames, atom_types, masses


def _coordinates_from_frames(frames: List[Atoms]) -> np.ndarray:
    coordinates = np.empty((len(frames), len(frames[0]), 3), dtype=np.float64)
    for frame_id, atoms in enumerate(frames):
        coordinates[frame_id, :, :] = np.asarray(atoms.coordinates, dtype=np.float64)
    coordinates.setflags(write=False)
    return coordinates


def _coordinate_cache_identity(manifest: "TrajectoryPoolManifest") -> Dict[str, Any]:
    return {
        "pool_sha256": str(manifest.sha256),
        "n_frames": int(manifest.n_frames),
        "natoms": int(manifest.natoms),
        "atom_types": list(manifest.atom_types),
        "masses": [float(value) for value in manifest.masses],
        "encoding_version": POOL_COORDINATE_ENCODING_VERSION,
        "parser_contract": POOL_COORDINATE_PARSER_CONTRACT,
        "dtype": np.dtype(np.float64).str,
        "byteorder": sys.byteorder,
        "platform": platform.system(),
        "machine": platform.machine(),
        "numpy_version": str(np.__version__),
    }


def _coordinate_cache_paths(
    campaign: Path, manifest: "TrajectoryPoolManifest"
) -> Dict[str, Path]:
    root = campaign_owned_path(campaign, POOL_COORDINATE_CACHE_SUBDIR)
    identity_dir = root / str(manifest.sha256)
    return {
        "root": root,
        "directory": identity_dir,
        "data": identity_dir / POOL_COORDINATE_CACHE_FILENAME,
        "manifest": identity_dir / POOL_COORDINATE_CACHE_MANIFEST_FILENAME,
        "lock": root / (str(manifest.sha256) + ".lock"),
        "building": root / (".building-" + str(manifest.sha256)),
    }


def _read_coordinate_cache(
    directory: Path, manifest: "TrajectoryPoolManifest"
) -> np.ndarray:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("trajectory coordinate cache directory is invalid")
    manifest_path = directory / POOL_COORDINATE_CACHE_MANIFEST_FILENAME
    data_path = directory / POOL_COORDINATE_CACHE_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("trajectory coordinate cache manifest is missing")
    if data_path.is_symlink() or not data_path.is_file():
        raise ValueError("trajectory coordinate cache data is missing")
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"), source=manifest_path
    )
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "identity",
        "data",
    }:
        raise ValueError("trajectory coordinate cache manifest fields are invalid")
    if payload.get("schema_version") != POOL_COORDINATE_CACHE_SCHEMA_VERSION:
        raise ValueError("trajectory coordinate cache schema is unsupported")
    if payload.get("identity") != _coordinate_cache_identity(manifest):
        raise ValueError("trajectory coordinate cache identity mismatch")
    data = payload.get("data")
    if not isinstance(data, dict) or set(data) != {
        "name",
        "size",
        "sha256",
        "shape",
        "dtype",
    }:
        raise ValueError("trajectory coordinate cache data record is invalid")
    if data.get("name") != POOL_COORDINATE_CACHE_FILENAME:
        raise ValueError("trajectory coordinate cache filename is invalid")
    if data_path.stat().st_size != int(data.get("size", -1)):
        raise ValueError("trajectory coordinate cache size mismatch")
    if sha256_file(data_path) != str(data.get("sha256") or ""):
        raise ValueError("trajectory coordinate cache hash mismatch")
    expected_shape = (int(manifest.n_frames), int(manifest.natoms), 3)
    if data.get("shape") != list(expected_shape):
        raise ValueError("trajectory coordinate cache shape record mismatch")
    if data.get("dtype") != np.dtype(np.float64).str:
        raise ValueError("trajectory coordinate cache dtype record mismatch")
    coordinates = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if coordinates.shape != expected_shape or coordinates.dtype != np.dtype(np.float64):
        raise ValueError("trajectory coordinate cache array contract mismatch")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("trajectory coordinate cache contains non-finite values")
    return coordinates


def _report_progress(
    callback: Optional[Callable[[str, Dict[str, Any]], None]],
    stage: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), dict(payload))
    except Exception:
        pass


def _validated_coordinates(
    path: Path,
    *,
    manifest: "TrajectoryPoolManifest",
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> np.ndarray:
    coordinates = np.empty(
        (int(manifest.n_frames), int(manifest.natoms), 3), dtype=np.float64
    )
    observed_frames = 0
    for frame_id, atoms in enumerate(iter_xyz_frames(path)):
        if frame_id >= int(manifest.n_frames):
            raise RuntimeError("pool manifest metadata does not match canonical pool.xyz")
        atom_types = tuple(atom.type for atom in atoms)
        masses = tuple(float(atom.mass) for atom in atoms)
        if atom_types != manifest.atom_types or masses != manifest.masses:
            raise RuntimeError("pool manifest metadata does not match canonical pool.xyz")
        frame_coordinates = np.asarray(atoms.coordinates, dtype=np.float64)
        if frame_coordinates.shape != (int(manifest.natoms), 3):
            raise RuntimeError("pool manifest metadata does not match canonical pool.xyz")
        if not np.all(np.isfinite(frame_coordinates)):
            raise RuntimeError("canonical pool.xyz contains non-finite coordinates")
        coordinates[frame_id, :, :] = frame_coordinates
        observed_frames += 1
        if observed_frames == 1 or observed_frames % 256 == 0:
            _report_progress(
                progress_callback,
                "trajectory_coordinates",
                completed=int(observed_frames),
                total=int(manifest.n_frames),
                cache_status="building",
            )
    if observed_frames != int(manifest.n_frames):
        raise RuntimeError("pool manifest metadata does not match canonical pool.xyz")
    coordinates.setflags(write=False)
    return coordinates


def _remove_derived_cache_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return
    retired = path.with_name(".invalid-" + uuid.uuid4().hex[:12])
    os.replace(path, retired)
    shutil.rmtree(retired, ignore_errors=True)


def _publish_coordinate_cache(
    campaign: Path,
    manifest: "TrajectoryPoolManifest",
    coordinates: np.ndarray,
) -> np.ndarray:
    paths = _coordinate_cache_paths(campaign, manifest)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _remove_derived_cache_path(paths["building"])
    paths["building"].mkdir()
    data_path = paths["building"] / POOL_COORDINATE_CACHE_FILENAME
    with data_path.open("xb") as handle:
        np.save(handle, np.asarray(coordinates, dtype=np.float64), allow_pickle=False)
        handle.flush()
        _fsync_file_descriptor(handle.fileno())
    cache_manifest = {
        "schema_version": POOL_COORDINATE_CACHE_SCHEMA_VERSION,
        "identity": _coordinate_cache_identity(manifest),
        "data": {
            "name": POOL_COORDINATE_CACHE_FILENAME,
            "size": int(data_path.stat().st_size),
            "sha256": sha256_file(data_path),
            "shape": [int(manifest.n_frames), int(manifest.natoms), 3],
            "dtype": np.dtype(np.float64).str,
        },
    }
    atomic_write_json(
        paths["building"] / POOL_COORDINATE_CACHE_MANIFEST_FILENAME,
        cache_manifest,
    )
    if paths["directory"].exists() or paths["directory"].is_symlink():
        _remove_derived_cache_path(paths["directory"])
    os.replace(paths["building"], paths["directory"])
    _fsync_parent_dir(paths["directory"])
    published = _read_coordinate_cache(paths["directory"], manifest)
    for child in paths["root"].iterdir():
        if (
            child != paths["directory"]
            and child.is_dir()
            and not child.is_symlink()
            and not child.name.startswith(".building-")
        ):
            _remove_derived_cache_path(child)
    return published


def _load_or_build_coordinate_cache(
    campaign: Path,
    canonical_path: Path,
    manifest: "TrajectoryPoolManifest",
    *,
    progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> np.ndarray:
    try:
        paths = _coordinate_cache_paths(campaign, manifest)
        if paths["directory"].exists() or paths["directory"].is_symlink():
            try:
                coordinates = _read_coordinate_cache(paths["directory"], manifest)
                _report_progress(
                    progress_callback,
                    "trajectory_coordinates",
                    completed=int(manifest.n_frames),
                    total=int(manifest.n_frames),
                    cache_status="hit",
                )
                return coordinates
            except (OSError, ValueError):
                pass
        paths["root"].mkdir(parents=True, exist_ok=True)
        if paths["lock"].is_symlink():
            raise ValueError("trajectory coordinate cache lock is symlinked")
        import portalocker

        try:
            with portalocker.Lock(str(paths["lock"]), mode="a", timeout=600):
                if paths["directory"].exists() or paths["directory"].is_symlink():
                    try:
                        coordinates = _read_coordinate_cache(
                            paths["directory"], manifest
                        )
                        _report_progress(
                            progress_callback,
                            "trajectory_coordinates",
                            completed=int(manifest.n_frames),
                            total=int(manifest.n_frames),
                            cache_status="hit",
                        )
                        return coordinates
                    except (OSError, ValueError):
                        _remove_derived_cache_path(paths["directory"])
                coordinates = _validated_coordinates(
                    canonical_path,
                    manifest=manifest,
                    progress_callback=progress_callback,
                )
                try:
                    published = _publish_coordinate_cache(
                        campaign, manifest, coordinates
                    )
                    _report_progress(
                        progress_callback,
                        "trajectory_coordinates",
                        completed=int(manifest.n_frames),
                        total=int(manifest.n_frames),
                        cache_status="published",
                    )
                    return published
                except (OSError, ValueError):
                    return coordinates
        except portalocker.exceptions.LockException:
            return _validated_coordinates(
                canonical_path,
                manifest=manifest,
                progress_callback=progress_callback,
            )
    except (OSError, ValueError):
        return _validated_coordinates(
            canonical_path,
            manifest=manifest,
            progress_callback=progress_callback,
        )


def _checked_unlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("pool transaction path must not be a symlink: " + str(path))
    if path.exists():
        if not path.is_file():
            raise ValueError("pool transaction path is not a regular file: " + str(path))
        path.unlink()
        _fsync_parent_dir(path)


def _transaction_paths(campaign: Path, token: str) -> Dict[str, Path]:
    if len(token) != 32 or any(ch not in "0123456789abcdef" for ch in token):
        raise ValueError("pool import transaction identifier is invalid")
    trajectory_dir = campaign_owned_path(campaign, POOL_SUBDIR)
    return {
        "canonical": campaign_owned_path(campaign, POOL_XYZ_FILENAME),
        "manifest": campaign_owned_path(
            campaign, POOL_SUBDIR / POOL_MANIFEST_FILENAME
        ),
        "transaction": campaign_owned_path(
            campaign, POOL_SUBDIR / POOL_IMPORT_TRANSACTION_FILENAME
        ),
        "staged": campaign_owned_path(
            campaign, ".pool.xyz.import-" + token + ".tmp"
        ),
        "pool_backup": campaign_owned_path(
            campaign, ".pool.xyz.backup-" + token
        ),
        "manifest_backup": campaign_owned_path(
            campaign,
            trajectory_dir / (".pool.manifest.backup-" + token + ".json"),
        ),
    }


def _recover_import_transaction(campaign: Path) -> None:
    transaction = campaign_owned_path(
        campaign, POOL_SUBDIR / POOL_IMPORT_TRANSACTION_FILENAME
    )
    if transaction.is_symlink():
        raise ValueError("pool import transaction must not be a symlink")
    if not transaction.exists():
        return
    if not transaction.is_file():
        raise ValueError("pool import transaction is not a regular file")
    payload = json.loads(transaction.read_text(encoding="utf-8"), source=transaction)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("pool import transaction has an unsupported schema")
    paths = _transaction_paths(campaign, str(payload.get("transaction_id") or ""))
    canonical = paths["canonical"]
    manifest = paths["manifest"]
    new_sha = str(payload.get("new_pool_sha256") or "")

    committed = False
    if canonical.is_file() and not canonical.is_symlink() and manifest.is_file() and not manifest.is_symlink():
        try:
            manifest_payload = json.loads(
                manifest.read_text(encoding="utf-8"), source=manifest
            )
            committed = (
                isinstance(manifest_payload, dict)
                and str(manifest_payload.get("sha256") or "") == new_sha
                and sha256_file(canonical) == new_sha
            )
        except (OSError, ValueError):
            committed = False

    if not committed:
        old_pool_exists = payload.get("old_pool_exists") is True
        old_manifest_exists = payload.get("old_manifest_exists") is True
        old_pool_sha = payload.get("old_pool_sha256")
        old_manifest_sha = payload.get("old_manifest_sha256")

        if paths["pool_backup"].is_file() and not paths["pool_backup"].is_symlink():
            _checked_unlink(canonical)
            os.replace(paths["pool_backup"], canonical)
            _fsync_parent_dir(canonical)
        elif old_pool_exists:
            if not canonical.is_file() or sha256_file(canonical) != old_pool_sha:
                raise RuntimeError("interrupted pool import cannot restore prior pool bytes")
        else:
            _checked_unlink(canonical)

        if paths["manifest_backup"].is_file() and not paths["manifest_backup"].is_symlink():
            _checked_unlink(manifest)
            os.replace(paths["manifest_backup"], manifest)
            _fsync_parent_dir(manifest)
        elif old_manifest_exists:
            if not manifest.is_file() or sha256_file(manifest) != old_manifest_sha:
                raise RuntimeError("interrupted pool import cannot restore prior manifest")
        else:
            _checked_unlink(manifest)

    for key in ("staged", "pool_backup", "manifest_backup"):
        _checked_unlink(paths[key])
    _checked_unlink(transaction)


@dataclass(frozen=True)
class TrajectoryPoolManifest:
    """Pin metadata for one trajectory pool.

    The SHA-256 of the canonical pool.xyz is the durable identity. Every
    iteration manifest emitted by the daemon includes this hash so the
    provenance chain back to the MD source is auditable end-to-end.
    """

    source_path: str          # original path; recorded but not authoritative
    canonical_path: str       # <campaign>/pool.xyz
    sha256: str               #of the canonical file
    n_frames: int
    natoms: int
    atom_types: Tuple[str, ...]
    masses: Tuple[float, ...]
    imported_iso: str = ""
    schema_version: int = POOL_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Lists are JSON-friendly; the dataclass uses tuples for immutability.
        d["atom_types"] = list(self.atom_types)
        d["masses"] = list(self.masses)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryPoolManifest":
        if not isinstance(data, dict):
            raise ValueError("pool manifest must be a JSON object")
        expected_fields = {
            "source_path",
            "canonical_path",
            "sha256",
            "n_frames",
            "natoms",
            "atom_types",
            "masses",
            "imported_iso",
            "schema_version",
        }
        if set(data) != expected_fields:
            raise ValueError(
                "pool manifest fields are invalid: expected "
                + repr(sorted(expected_fields))
            )
        schema = data.get("schema_version")
        if isinstance(schema, bool) or not isinstance(schema, int):
            raise ValueError("pool manifest schema_version must be an integer")
        if schema != POOL_SCHEMA_VERSION:
            raise ValueError(
                "pool manifest schema_version " + str(schema)
                + " != " + str(POOL_SCHEMA_VERSION)
            )
        required_str = ("source_path", "canonical_path", "sha256", "imported_iso")
        for key in required_str:
            if not isinstance(data.get(key), str) or not data[key]:
                raise ValueError("missing or non-string manifest field: " + key)
        digest = data["sha256"]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("pool manifest sha256 is malformed")
        try:
            imported_at = datetime.fromisoformat(data["imported_iso"])
        except ValueError as exc:
            raise ValueError("pool manifest imported_iso is malformed") from exc
        if imported_at.tzinfo is None:
            raise ValueError("pool manifest imported_iso must include a timezone")
        required_int = ("n_frames", "natoms")
        for key in required_int:
            if (
                isinstance(data.get(key), bool)
                or not isinstance(data.get(key), int)
                or data[key] <= 0
            ):
                raise ValueError("missing or non-int manifest field: " + key)
        atom_types = data.get("atom_types")
        masses = data.get("masses")
        if (
            not isinstance(atom_types, list)
            or not all(isinstance(a, str) and a for a in atom_types)
        ):
            raise ValueError("atom_types must be a list of strings")
        if (
            not isinstance(masses, list)
            or not all(
                not isinstance(mass, bool)
                and isinstance(mass, (int, float))
                and np.isfinite(float(mass))
                and float(mass) > 0.0
                for mass in masses
            )
        ):
            raise ValueError("masses must be a list of numbers")
        if len(atom_types) != data["natoms"] or len(masses) != data["natoms"]:
            raise ValueError("atom_types / masses length disagrees with natoms")
        return cls(
            source_path=data["source_path"],
            canonical_path=data["canonical_path"],
            sha256=data["sha256"],
            n_frames=int(data["n_frames"]),
            natoms=int(data["natoms"]),
            atom_types=tuple(atom_types),
            masses=tuple(float(m) for m in masses),
            imported_iso=data["imported_iso"],
            schema_version=schema,
        )


class TrajectoryPool:
    """Random access view over an immutable per-campaign trajectory.

    Construct via :meth: import_from (first time) or :meth: load (every
    subsequent daemon start). Frames are accessed by stable
    frame_id in range(n_frames).
    """

    def __init__(
        self,
        manifest: TrajectoryPoolManifest,
        frames: Union[List[Atoms], np.ndarray],
    ) -> None:
        if isinstance(frames, np.ndarray):
            coordinates = frames
        else:
            coordinates = _coordinates_from_frames(frames)
        expected_shape = (int(manifest.n_frames), int(manifest.natoms), 3)
        if coordinates.shape != expected_shape:
            raise ValueError(
                "manifest n_frames=" + str(manifest.n_frames)
                + " disagrees with coordinate shape=" + repr(coordinates.shape)
            )
        if coordinates.dtype != np.dtype(np.float64):
            raise ValueError("trajectory coordinates must use float64 encoding")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("trajectory coordinates contain non-finite values")
        self._manifest = manifest
        self._coordinates = coordinates

    # --- accessors ----------------------------------------------------

    @property
    def manifest(self) -> TrajectoryPoolManifest:
        return self._manifest

    @property
    def canonical_path(self) -> Path:
        return Path(self._manifest.canonical_path)

    @property
    def sha256(self) -> str:
        return self._manifest.sha256

    def n_frames(self) -> int:
        return self._manifest.n_frames

    def __len__(self) -> int:
        return self.n_frames()

    def __getitem__(self, frame_id):
        if isinstance(frame_id, slice):
            return [self.frame(index) for index in range(*frame_id.indices(len(self)))]
        return self.frame(frame_id)

    def frame(self, frame_id: int) -> Atoms:
        """Return a detached copy of the frame at the stable frame ID."""
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)):
            raise TypeError("frame_id must be an exact integer")
        frame_index = int(frame_id)
        if not 0 <= frame_index < self.n_frames():
            raise IndexError("frame_id " + str(frame_id) + " out of range [0, " + str(self.n_frames()) + ")")
        coordinates = self._coordinates[frame_index]
        return Atoms(
            [
                Atom(atom_type, float(x), float(y), float(z))
                for atom_type, (x, y, z) in zip(
                    self._manifest.atom_types, coordinates
                )
            ]
        )

    def frame_ids(self) -> range:
        """Return the inclusive range of all stable frame IDs."""
        return range(self.n_frames())

    def coordinates_view(self) -> np.ndarray:
        """Return the immutable, frame-ordered coordinate array."""
        view = np.asarray(self._coordinates, dtype=np.float64).view()
        view.setflags(write=False)
        return view

    def frame_coordinates(self, frame_id: int) -> np.ndarray:
        """Return an immutable coordinate view for one stable frame ID."""
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)):
            raise TypeError("frame_id must be an exact integer")
        frame_index = int(frame_id)
        if not 0 <= frame_index < self.n_frames():
            raise IndexError("frame_id is outside the trajectory pool")
        view = np.asarray(self._coordinates[frame_index], dtype=np.float64).view()
        view.setflags(write=False)
        return view

    def to_atoms_list(self) -> List[Atoms]:
        """Return detached frame copies that cannot mutate the pool."""
        return [self.frame(frame_id) for frame_id in self.frame_ids()]

    # --- import + load -----------------------------------------------

    @classmethod
    def import_from(
        cls,
        source: Union[str, Path],
        campaign_dir: Union[str, Path],
        *,
        overwrite: bool = False,
    ) -> "TrajectoryPool":
        """Copy source into the canonical pool path under campaign_dir
        and write the manifest. Refuses to overwrite an existing manifest
        unless overwrite=True (explicit user opt-in only).

        The import is deliberately verbatim: every source frame is pinned in
        the campaign pool and downstream selection/safety gates decide which
        frames are useful.
        """
        from ..operator_paths import reject_operator_input_symlinks

        source = reject_operator_input_symlinks(Path(source))
        if not source.is_file():
            raise FileNotFoundError("trajectory source does not exist: " + str(source))
        campaign_dir = Path(campaign_dir)
        target_dir = campaign_owned_path(campaign_dir, POOL_SUBDIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        _recover_import_transaction(campaign_dir)
        manifest_path = campaign_owned_path(
            campaign_dir, POOL_SUBDIR / POOL_MANIFEST_FILENAME
        )
        canonical_path = campaign_owned_path(campaign_dir, POOL_XYZ_FILENAME)
        if manifest_path.exists() and not manifest_path.is_file():
            raise ValueError("pool manifest path is not a regular file")
        if canonical_path.exists() and not canonical_path.is_file():
            raise ValueError("campaign pool path is not a regular file")
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                "pool manifest already exists at " + str(manifest_path)
                + " -- refusing to overwrite. Start a new campaign or pass overwrite=True."
            )
        source_is_canonical = source.resolve() == canonical_path.resolve(strict=False)
        if canonical_path.exists() and not overwrite and not source_is_canonical:
            raise FileExistsError(
                "campaign pool already exists at "
                + str(canonical_path)
                + " -- refusing to overwrite"
            )

        token = uuid.uuid4().hex
        paths = _transaction_paths(campaign_dir, token)
        staged = paths["staged"]
        transaction_written = False
        try:
            with source.open("rb") as input_handle, staged.open("xb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle)
                output_handle.flush()
                _fsync_file_descriptor(output_handle.fileno())
            atoms_list, atom_types, masses = _validated_frames(
                staged, source_label=source
            )
            sha = sha256_file(staged)
            manifest = TrajectoryPoolManifest(
                source_path=str(source.resolve()),
                canonical_path=str(canonical_path.resolve(strict=False)),
                sha256=sha,
                n_frames=len(atoms_list),
                natoms=len(atoms_list[0]),
                atom_types=atom_types,
                masses=masses,
                imported_iso=datetime.now(timezone.utc).isoformat(),
            )
            atomic_write_json(
                paths["transaction"],
                {
                    "schema_version": POOL_IMPORT_TRANSACTION_SCHEMA_VERSION,
                    "transaction_id": token,
                    "new_pool_sha256": sha,
                    "old_pool_exists": canonical_path.is_file(),
                    "old_pool_sha256": (
                        sha256_file(canonical_path) if canonical_path.is_file() else None
                    ),
                    "old_manifest_exists": manifest_path.is_file(),
                    "old_manifest_sha256": (
                        sha256_file(manifest_path) if manifest_path.is_file() else None
                    ),
                },
            )
            transaction_written = True
            if canonical_path.is_file():
                os.replace(canonical_path, paths["pool_backup"])
                _fsync_parent_dir(canonical_path)
            if manifest_path.is_file():
                os.replace(manifest_path, paths["manifest_backup"])
                _fsync_parent_dir(manifest_path)
            os.replace(staged, canonical_path)
            _fsync_parent_dir(canonical_path)
            atomic_write_json(manifest_path, manifest.to_dict())
            if sha256_file(canonical_path) != sha:
                raise RuntimeError("published campaign pool failed SHA verification")
            _recover_import_transaction(campaign_dir)
        except BaseException:
            if transaction_written:
                try:
                    _recover_import_transaction(campaign_dir)
                except Exception as recovery_exc:
                    raise RuntimeError(
                        "pool import failed and rollback could not be completed"
                    ) from recovery_exc
            else:
                _checked_unlink(staged)
            raise
        coordinates = _coordinates_from_frames(atoms_list)
        try:
            coordinates = _publish_coordinate_cache(
                campaign_dir, manifest, coordinates
            )
        except (OSError, ValueError):
            pass
        return cls(manifest, coordinates)

    @classmethod
    def load(
        cls,
        campaign_dir: Union[str, Path],
        *,
        progress_callback: Optional[
            Callable[[str, Dict[str, Any]], None]
        ] = None,
    ) -> "TrajectoryPool":
        """Load ``<campaign_dir>/pool.xyz`` using its daemon manifest.

        Re-verifies the SHA-256 of the canonical pool against the manifest and raises
        if drift is detected (someone touched the file under us)."""
        campaign_dir = Path(campaign_dir)
        _report_progress(
            progress_callback,
            "trajectory_authority",
            completed=0,
            total=1,
        )
        _recover_import_transaction(campaign_dir)
        manifest_path = campaign_owned_path(
            campaign_dir, POOL_SUBDIR / POOL_MANIFEST_FILENAME
        )
        canonical_path = campaign_owned_path(campaign_dir, POOL_XYZ_FILENAME)
        if manifest_path.is_symlink():
            raise RuntimeError("pool manifest must not be a symlink: " + str(manifest_path))
        if canonical_path.is_symlink():
            raise RuntimeError("canonical pool xyz must not be a symlink: " + str(canonical_path))
        if not manifest_path.is_file():
            raise FileNotFoundError("pool manifest not found at " + str(manifest_path))
        if not canonical_path.is_file():
            raise FileNotFoundError("canonical pool xyz not found at " + str(canonical_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = TrajectoryPoolManifest.from_dict(data)
        if Path(manifest.canonical_path).resolve() != canonical_path.resolve():
            raise RuntimeError(
                "pool manifest canonical_path does not identify campaign pool.xyz: "
                + str(manifest.canonical_path)
            )
        on_disk_sha = sha256_file(canonical_path)
        if on_disk_sha != manifest.sha256:
            raise RuntimeError(
                "pool drift detected: canonical pool.xyz SHA "
                + on_disk_sha + " != manifest " + manifest.sha256
                + " -- the trajectory was modified out-of-band. Refusing to load."
            )
        _report_progress(
            progress_callback,
            "trajectory_authority",
            completed=1,
            total=1,
        )
        _report_progress(
            progress_callback,
            "trajectory_coordinates",
            completed=0,
            total=int(manifest.n_frames),
        )
        coordinates = _load_or_build_coordinate_cache(
            campaign_dir,
            canonical_path,
            manifest,
            progress_callback=progress_callback,
        )
        _report_progress(
            progress_callback,
            "trajectory_coordinates",
            completed=int(manifest.n_frames),
            total=int(manifest.n_frames),
        )
        return cls(manifest, coordinates)
