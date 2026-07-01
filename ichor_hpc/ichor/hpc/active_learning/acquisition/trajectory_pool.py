"""Trajectory pool: the immutable, content-addressed/hashed input to every campaign.

The MD / metadynamics trajectory is the campaign most precious input.
This module wraps it in an artefact with stable frame IDs, a SHA-256
content hash, and an atomic manifest that every iteration manifest can
transitively prove descendance from.

Layout on disk under <campaign_dir>:

    .DATA/TRAJECTORY/
        pool.xyz                # canonical copy of the operator MD trajectory
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

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from ichor.core.atoms import Atoms
from ichor.core.files.xyz import Trajectory

from ..daemon.state import atomic_write_json
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
POOL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrajectoryPoolManifest:
    """Pin metadata for one trajectory pool.

    The SHA-256 of the canonical pool.xyz is the durable identity. Every
    iteration manifest emitted by the daemon includes this hash so the
    provenance chain back to the MD source is auditable end-to-end.
    """

    source_path: str          # original path; recorded but not authoritative
    canonical_path: str       #<campaign>/.DATA/TRAJECTORY/pool.xyz
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
        schema = int(data.get("schema_version", -1))
        if schema != POOL_SCHEMA_VERSION:
            raise ValueError(
                "pool manifest schema_version " + str(schema)
                + " != " + str(POOL_SCHEMA_VERSION)
            )
        required_str = ("source_path", "canonical_path", "sha256", "imported_iso")
        for key in required_str:
            if not isinstance(data.get(key), str):
                raise ValueError("missing or non-string manifest field: " + key)
        required_int = ("n_frames", "natoms")
        for key in required_int:
            if not isinstance(data.get(key), int):
                raise ValueError("missing or non-int manifest field: " + key)
        atom_types = data.get("atom_types")
        masses = data.get("masses")
        if not isinstance(atom_types, list) or not all(isinstance(a, str) for a in atom_types):
            raise ValueError("atom_types must be a list of strings")
        if not isinstance(masses, list) or not all(isinstance(m, (int, float)) for m in masses):
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

    def frame(self, frame_id: int) -> Atoms:
        """Return the Atoms object at the given stable frame_id."""
        if not 0 <= int(frame_id) < self.n_frames():
            raise IndexError("frame_id " + str(frame_id) + " out of range [0, " + str(self.n_frames()) + ")")
        return self._atoms[int(frame_id)]

    def frame_ids(self) -> range:
        """Return the inclusive range of all stable frame IDs."""
        return range(self.n_frames())

    def to_atoms_list(self) -> List[Atoms]:
        """Return a *new* list of Atoms (callers may mutate without affecting the pool)."""
        return list(self._atoms)

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
        unless overwrite=True (operator opt-in only).

        The import is deliberately verbatim: every source frame is pinned in
        the campaign pool and downstream selection/safety gates decide which
        frames are useful.
        """
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError("trajectory source does not exist: " + str(source))
        campaign_dir = Path(campaign_dir)
        target_dir = campaign_dir / POOL_SUBDIR
        manifest_path = target_dir / POOL_MANIFEST_FILENAME
        canonical_path = target_dir / POOL_XYZ_FILENAME
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(
                "pool manifest already exists at " + str(manifest_path)
                + " -- refusing to overwrite. Start a new campaign or pass overwrite=True."
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        # Write to a temp path and atomically rename, so a crash mid-write
        # cannot leave the manifest pinning a half-written pool.xyz.
        tmp_canonical = canonical_path.with_name(canonical_path.name + ".tmp")
        shutil.copyfile(source, tmp_canonical)
        os.replace(str(tmp_canonical), str(canonical_path))
        sha = sha256_file(canonical_path)
        traj = Trajectory(canonical_path)
        traj.read()
        atoms_list: List[Atoms] = [atoms.copy() for atoms in traj]
        if not atoms_list:
            raise ValueError("trajectory has zero frames: " + str(source))
        head = atoms_list[0]
        atom_types = tuple(a.type for a in head)
        masses = tuple(float(a.mass) for a in head)
        #sanity-check that every frame has the same atom layout.
        for i, atoms in enumerate(atoms_list[1:], start=1):
            this_types = tuple(a.type for a in atoms)
            if this_types != atom_types:
                raise ValueError(
                    "frame " + str(i) + " atom types " + repr(this_types)
                    + " disagree with frame 0 " + repr(atom_types)
                )
        manifest = TrajectoryPoolManifest(
            source_path=str(source.resolve()),
            canonical_path=str(canonical_path),
            sha256=sha,
            n_frames=len(atoms_list),
            natoms=len(head),
            atom_types=atom_types,
            masses=masses,
            imported_iso=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(manifest_path, manifest.to_dict())
        return cls(manifest, atoms_list)

    @classmethod
    def load(cls, campaign_dir: Union[str, Path]) -> "TrajectoryPool":
        """Load the pool from <campaign_dir>/.DATA/TRAJECTORY/. Re-verifies
        the SHA-256 of the canonical copy against the manifest and raises
        if drift is detected (someone touched the file under us)."""
        campaign_dir = Path(campaign_dir)
        manifest_path = campaign_dir / POOL_SUBDIR / POOL_MANIFEST_FILENAME
        canonical_path = campaign_dir / POOL_SUBDIR / POOL_XYZ_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError("pool manifest not found at " + str(manifest_path))
        if not canonical_path.is_file():
            raise FileNotFoundError("canonical pool xyz not found at " + str(canonical_path))
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = TrajectoryPoolManifest.from_dict(data)
        on_disk_sha = sha256_file(canonical_path)
        if on_disk_sha != manifest.sha256:
            raise RuntimeError(
                "pool drift detected: canonical pool.xyz SHA "
                + on_disk_sha + " != manifest " + manifest.sha256
                + " -- the trajectory was modified out-of-band. Refusing to load."
            )
        traj = Trajectory(canonical_path)
        traj.read()
        atoms_list = [atoms.copy() for atoms in traj]
        return cls(manifest, atoms_list)
