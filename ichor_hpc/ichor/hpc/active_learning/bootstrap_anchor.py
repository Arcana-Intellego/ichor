"""Bootstrap anchor handling for model-0 initial training.

When enabled, ``anchor.xyz`` provides operator-selected geometries that must
be present in the initial labelled set and forced into the FEREBUS training
split.  Anchors are deliberately not assigned to internal or external
validation, because those rows are intended to act as basis centres for the
first predictive mean rather than as holdout scoring rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ichor.core.atoms import Atoms
from ichor.core.files.xyz import Trajectory

from .daemon.state import atomic_write_json
from .sampling.descriptors import mass_weighted_rmsd


ANCHOR_XYZ_FILENAME = "anchor.xyz"
BOOTSTRAP_ANCHOR_MANIFEST_FILENAME = "bootstrap_anchor.json"
BOOTSTRAP_ANCHOR_SCHEMA_VERSION = 1
ANCHOR_DUPLICATE_RMSD_ANGSTROM = 1.0e-8


@dataclass(frozen=True)
class BootstrapAnchorPlan:
    enabled: bool
    anchor_path: str
    n_anchor: int
    bootstrap_total_size: int
    bootstrap_training_size: int
    bootstrap_internal_validation_size: int
    bootstrap_external_validation_size: int
    pool_total_needed: int
    pool_train_needed: int
    pool_internal_validation_needed: int
    pool_external_validation_needed: int
    excluded_pool_frame_ids: tuple[int, ...] = ()
    duplicate_pool_frame_ids_by_anchor: tuple[tuple[int, ...], ...] = ()
    atom_types: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": BOOTSTRAP_ANCHOR_SCHEMA_VERSION,
            "enabled": bool(self.enabled),
            "anchor_path": str(self.anchor_path),
            "n_anchor": int(self.n_anchor),
            "bootstrap_total_size": int(self.bootstrap_total_size),
            "bootstrap_training_size": int(self.bootstrap_training_size),
            "bootstrap_internal_validation_size": int(
                self.bootstrap_internal_validation_size
            ),
            "bootstrap_external_validation_size": int(
                self.bootstrap_external_validation_size
            ),
            "pool_total_needed": int(self.pool_total_needed),
            "pool_train_needed": int(self.pool_train_needed),
            "pool_internal_validation_needed": int(
                self.pool_internal_validation_needed
            ),
            "pool_external_validation_needed": int(
                self.pool_external_validation_needed
            ),
            "excluded_pool_frame_ids": [
                int(i) for i in self.excluded_pool_frame_ids
            ],
            "duplicate_pool_frame_ids_by_anchor": [
                [int(i) for i in ids]
                for ids in self.duplicate_pool_frame_ids_by_anchor
            ],
            "atom_types": [str(t) for t in self.atom_types],
        }


def anchor_xyz_path(campaign_dir: str | Path) -> Path:
    return Path(campaign_dir) / ANCHOR_XYZ_FILENAME


def bootstrap_anchor_manifest_path(campaign_dir: str | Path) -> Path:
    return (
        Path(campaign_dir)
        / ".DATA"
        / "ACTIVE_LEARNING"
        / BOOTSTRAP_ANCHOR_MANIFEST_FILENAME
    )


def load_anchor_frames(campaign_dir: str | Path) -> List[Atoms]:
    path = anchor_xyz_path(campaign_dir)
    if path.is_symlink():
        raise ValueError(
            "point_allocation.anchor refuses symlinked anchor.xyz: " + str(path)
        )
    if not path.is_file():
        raise FileNotFoundError(
            "point_allocation.anchor is true but anchor.xyz is missing: " + str(path)
        )
    try:
        traj = Trajectory(path)
        traj.read()
        frames = [atoms.copy() for atoms in traj]
    except Exception as exc:
        raise ValueError("failed to read bootstrap anchor.xyz: " + str(path)) from exc
    if not frames:
        raise ValueError("bootstrap anchor.xyz contains no geometries: " + str(path))
    return frames


def _atom_types(atoms: Atoms) -> tuple[str, ...]:
    return tuple(str(t) for t in atoms.types_extended)


def _validate_geometry(frame: Atoms, *, label: str, expected_atom_types: Sequence[str]) -> None:
    atom_types = _atom_types(frame)
    expected = tuple(str(t) for t in expected_atom_types)
    if atom_types != expected:
        raise ValueError(
            label
            + " atom order/type mismatch: expected "
            + repr(list(expected))
            + ", got "
            + repr(list(atom_types))
        )
    coords = np.asarray(frame.coordinates, dtype=float)
    if coords.shape != (len(expected), 3):
        raise ValueError(
            label
            + " coordinate shape mismatch: expected "
            + repr((len(expected), 3))
            + ", got "
            + repr(tuple(coords.shape))
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError(label + " contains non-finite coordinates")


def _duplicate_anchor_pairs(anchor_frames: Sequence[Atoms]) -> List[tuple[int, int]]:
    out: List[tuple[int, int]] = []
    for i in range(len(anchor_frames)):
        for j in range(i + 1, len(anchor_frames)):
            if (
                float(mass_weighted_rmsd(anchor_frames[i], anchor_frames[j]))
                <= ANCHOR_DUPLICATE_RMSD_ANGSTROM
            ):
                out.append((i, j))
    return out


def _pool_duplicates(
    anchor_frames: Sequence[Atoms],
    pool_frames: Sequence[Atoms],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    by_anchor: List[tuple[int, ...]] = []
    excluded = set()
    for anchor in anchor_frames:
        matches: List[int] = []
        for frame_id, pool_frame in enumerate(pool_frames):
            if (
                float(mass_weighted_rmsd(anchor, pool_frame))
                <= ANCHOR_DUPLICATE_RMSD_ANGSTROM
            ):
                matches.append(int(frame_id))
                excluded.add(int(frame_id))
        by_anchor.append(tuple(matches))
    return tuple(sorted(excluded)), tuple(by_anchor)


def plan_bootstrap_anchors(
    campaign_dir: str | Path,
    config: Any,
    *,
    pool_frames: Optional[Sequence[Atoms]] = None,
) -> tuple[BootstrapAnchorPlan, List[Atoms]]:
    """Return the anchor plan and loaded anchor frames.

    The plan is anchor-aware even when anchors are disabled so callers can use
    the same fields for pool-feasibility maths and manifest diagnostics.
    """
    allocation = config.point_allocation
    initial_n = int(allocation.bootstrap_total_size)
    planned_train = int(allocation.bootstrap_training_size)
    planned_internal = int(allocation.bootstrap_internal_validation_size)
    external_n = int(allocation.bootstrap_external_validation_size)

    pool_frames_list = list(pool_frames or [])
    expected_atom_types: tuple[str, ...] = ()
    if pool_frames_list:
        expected_atom_types = _atom_types(pool_frames_list[0])
        for idx, frame in enumerate(pool_frames_list):
            _validate_geometry(
                frame,
                label="pool frame " + str(idx),
                expected_atom_types=expected_atom_types,
            )

    anchor_frames: List[Atoms] = []
    excluded_pool_ids: tuple[int, ...] = ()
    duplicate_by_anchor: tuple[tuple[int, ...], ...] = ()
    if bool(getattr(config.point_allocation, "anchor", False)):
        anchor_frames = load_anchor_frames(campaign_dir)
        if not expected_atom_types:
            expected_atom_types = _atom_types(anchor_frames[0])
        for idx, frame in enumerate(anchor_frames):
            _validate_geometry(
                frame,
                label="anchor frame " + str(idx),
                expected_atom_types=expected_atom_types,
            )
        duplicate_pairs = _duplicate_anchor_pairs(anchor_frames)
        if duplicate_pairs:
            raise ValueError(
                "bootstrap anchor.xyz contains duplicate geometries at "
                + repr(duplicate_pairs)
            )
        excluded_pool_ids, duplicate_by_anchor = _pool_duplicates(
            anchor_frames,
            pool_frames_list,
        )

    n_anchor = len(anchor_frames)
    if n_anchor > planned_train:
        raise ValueError(
            "point_allocation.anchor has "
            + str(n_anchor)
            + " geometries but the planned initial FEREBUS training split "
            + "can hold only "
            + str(planned_train)
            + " rows; reduce anchor.xyz or increase "
            + "point_allocation.bootstrap_training_size"
        )
    pool_total_needed = initial_n - n_anchor
    available_pool = len(pool_frames_list) - len(excluded_pool_ids)
    if pool_frames is not None and pool_total_needed > available_pool:
        raise ValueError(
            "bootstrap requires "
            + str(pool_total_needed)
            + " pool geometries after anchors, but only "
            + str(available_pool)
            + " non-anchor pool geometries are available"
        )
    plan = BootstrapAnchorPlan(
        enabled=bool(getattr(config.point_allocation, "anchor", False)),
        anchor_path=str(anchor_xyz_path(campaign_dir).resolve(strict=False)),
        n_anchor=int(n_anchor),
        bootstrap_total_size=int(initial_n),
        bootstrap_training_size=int(planned_train),
        bootstrap_internal_validation_size=int(planned_internal),
        bootstrap_external_validation_size=int(external_n),
        pool_total_needed=int(pool_total_needed),
        pool_train_needed=int(planned_train - n_anchor),
        pool_internal_validation_needed=int(planned_internal),
        pool_external_validation_needed=int(external_n),
        excluded_pool_frame_ids=tuple(int(i) for i in excluded_pool_ids),
        duplicate_pool_frame_ids_by_anchor=duplicate_by_anchor,
        atom_types=expected_atom_types,
    )
    return plan, anchor_frames


def write_bootstrap_anchor_manifest(
    campaign_dir: str | Path,
    plan: BootstrapAnchorPlan,
    *,
    selected_pool_frame_ids: Sequence[int],
    phase_a_sample_xyz: str | Path,
    phase_a_index_path: str | Path,
) -> Path:
    payload = plan.to_dict()
    payload["selected_pool_frame_ids"] = [
        int(i) for i in selected_pool_frame_ids
    ]
    payload["phase_a_sample_xyz"] = str(Path(phase_a_sample_xyz).resolve(strict=False))
    payload["phase_a_index_path"] = str(Path(phase_a_index_path).resolve(strict=False))
    path = bootstrap_anchor_manifest_path(campaign_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def read_bootstrap_anchor_manifest(campaign_dir: str | Path) -> Dict[str, Any]:
    path = bootstrap_anchor_manifest_path(campaign_dir)
    if not path.is_file():
        raise FileNotFoundError("bootstrap anchor manifest missing: " + str(path))
    import json

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("bootstrap anchor manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("bootstrap anchor manifest must be a JSON object")
    if int(data.get("schema_version", -1)) != BOOTSTRAP_ANCHOR_SCHEMA_VERSION:
        raise ValueError("unsupported bootstrap anchor manifest schema: " + str(path))
    return data
