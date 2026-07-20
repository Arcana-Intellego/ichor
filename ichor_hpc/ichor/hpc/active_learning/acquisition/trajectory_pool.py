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
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from ichor.core.atoms import Atoms
from ichor.core.files.xyz import Trajectory

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
]


POOL_SUBDIR = Path(".DATA") / "TRAJECTORY"
POOL_XYZ_FILENAME = "pool.xyz"
POOL_MANIFEST_FILENAME = "pool.manifest.json"
POOL_IMPORT_TRANSACTION_FILENAME = "pool.import.transaction.json"
POOL_SCHEMA_VERSION = 2
POOL_IMPORT_TRANSACTION_SCHEMA_VERSION = 1


def _validated_frames(path: Path, *, source_label: Path) -> tuple:
    trajectory = Trajectory(path)
    trajectory.read()
    frames = [atoms.copy() for atoms in trajectory]
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

    def __init__(self, manifest: TrajectoryPoolManifest, atoms_list: List[Atoms]) -> None:
        if manifest.n_frames != len(atoms_list):
            raise ValueError(
                "manifest n_frames=" + str(manifest.n_frames)
                + " disagrees with loaded list length=" + str(len(atoms_list))
            )
        self._manifest = manifest
        self._atoms: List[Atoms] = atoms_list

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
        return self._atoms[frame_index].copy()

    def frame_ids(self) -> range:
        """Return the inclusive range of all stable frame IDs."""
        return range(self.n_frames())

    def to_atoms_list(self) -> List[Atoms]:
        """Return detached frame copies that cannot mutate the pool."""
        return [atoms.copy() for atoms in self._atoms]

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
        return cls(manifest, atoms_list)

    @classmethod
    def load(cls, campaign_dir: Union[str, Path]) -> "TrajectoryPool":
        """Load ``<campaign_dir>/pool.xyz`` using its daemon manifest.

        Re-verifies the SHA-256 of the canonical pool against the manifest and raises
        if drift is detected (someone touched the file under us)."""
        campaign_dir = Path(campaign_dir)
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
        atoms_list, atom_types, masses = _validated_frames(
            canonical_path, source_label=canonical_path
        )
        if (
            len(atoms_list) != manifest.n_frames
            or len(atoms_list[0]) != manifest.natoms
            or atom_types != manifest.atom_types
            or masses != manifest.masses
        ):
            raise RuntimeError("pool manifest metadata does not match canonical pool.xyz")
        return cls(manifest, atoms_list)
