"""Finite reserve-only replacement sampling for incomplete QM allocations."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ichor.core.atoms import Atom, Atoms

from .daemon.state import atomic_write_json
from .point_allocation import (
    allocate_replacements,
    allocation_manifest_sha256,
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
)


REPLACEMENT_SAMPLE_FILENAME = "REPLACEMENT_SAMPLE.json"
REPLACEMENT_SAMPLE_SCHEMA_VERSION = 1


def replacement_round_dir(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
) -> Path:
    bucket = "initial" if str(context) == "bootstrap" else "iter_" + str(int(iteration))
    return (
        Path(campaign_dir)
        / ".DATA"
        / "STAGING"
        / bucket
        / ("replacement_round_" + str(int(replacement_round)).zfill(4))
    )


def replacement_sample_manifest_path(round_dir: str | Path) -> Path:
    return Path(round_dir) / REPLACEMENT_SAMPLE_FILENAME


def _write_xyz(frames: Sequence[Atoms], path: Path) -> None:
    lines: List[str] = []
    for index, frame in enumerate(frames):
        lines.append(str(len(frame)))
        lines.append("replacement frame " + str(index))
        for atom in frame:
            coordinates = (float(atom.x), float(atom.y), float(atom.z))
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("replacement geometry contains non-finite coordinates")
            lines.append(
                "{symbol} {x:.12f} {y:.12f} {z:.12f}".format(
                    symbol=atom.type,
                    x=coordinates[0],
                    y=coordinates[1],
                    z=coordinates[2],
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _active_frame(attempt: Mapping[str, Any]) -> Atoms:
    result_path = Path(str(attempt.get("result_json") or ""))
    if not result_path.is_file():
        raise FileNotFoundError("replacement ARIADNE result is missing: " + str(result_path))
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("replacement ARIADNE result is unreadable: " + str(result_path)) from exc
    atom_types = result.get("atom_types")
    coordinates = result.get("final_coordinates")
    if not isinstance(atom_types, list) or not isinstance(coordinates, list):
        raise ValueError("replacement ARIADNE result lacks final geometry")
    if len(atom_types) == 0 or len(atom_types) != len(coordinates):
        raise ValueError("replacement ARIADNE result geometry shape is invalid")
    atoms = []
    for atom_type, coordinate in zip(atom_types, coordinates):
        if not isinstance(coordinate, list) or len(coordinate) != 3:
            raise ValueError("replacement ARIADNE coordinate must contain three values")
        values = [float(coordinate[index]) for index in range(3)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("replacement ARIADNE geometry contains non-finite coordinates")
        if not str(atom_type).strip():
            raise ValueError("replacement ARIADNE geometry contains an empty atom type")
        atoms.append(
            Atom(
                str(atom_type),
                values[0],
                values[1],
                values[2],
            )
        )
    return Atoms(atoms)


def _bootstrap_frames(campaign_dir: Path, attempts: Sequence[Mapping[str, Any]]) -> List[Atoms]:
    from .acquisition.trajectory_pool import TrajectoryPool

    frames = TrajectoryPool.load(campaign_dir).to_atoms_list()
    selected: List[Atoms] = []
    for attempt in attempts:
        frame_id = attempt.get("frame_id")
        if isinstance(frame_id, bool) or frame_id is None:
            raise ValueError("bootstrap replacement candidate lacks a frame_id")
        index = int(frame_id)
        if index < 0 or index >= len(frames):
            raise ValueError("bootstrap replacement frame_id is outside the pool")
        selected.append(frames[index].copy())
    return selected


def prepare_replacement_round(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    allocation_path = point_allocation_path(
        campaign,
        context=context,
        iteration=int(iteration),
    )
    current = read_point_allocation(allocation_path)
    if bool((current.get("summary") or {}).get("complete", False)):
        raise ValueError("point allocation is already complete")
    existing_pending = pending_attempts(current)
    if existing_pending:
        pending_rounds = {int(attempt.get("round", -1)) for attempt in existing_pending}
        if pending_rounds != {int(replacement_round)}:
            raise ValueError(
                "point allocation already has pending candidates for replacement round(s) "
                + repr(sorted(pending_rounds))
            )
        updated = current
    else:
        updated = allocate_replacements(
            allocation_path,
            replacement_round=int(replacement_round),
            expected_generation=int(current.get("generation", 0)),
        )
    attempts = [
        attempt for attempt in pending_attempts(updated)
        if int(attempt.get("round", -1)) == int(replacement_round)
    ]
    if not attempts:
        raise ValueError("replacement round allocated no candidates")
    if str(context) == "bootstrap":
        frames = _bootstrap_frames(campaign, attempts)
    elif str(context) == "active":
        frames = [_active_frame(attempt) for attempt in attempts]
    else:
        raise ValueError("replacement context must be bootstrap or active")

    round_dir = replacement_round_dir(
        campaign,
        context=context,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    round_dir.mkdir(parents=True, exist_ok=True)
    sample_path = round_dir / "replacement-SAMPLE.xyz"
    _write_xyz(frames, sample_path)
    records = []
    target_total = int(updated["targets"]["total"])
    for sample_index, attempt in enumerate(attempts):
        reserve_rank = int(attempt.get("reserve_rank", sample_index))
        pointdir_index = target_total + reserve_rank
        if pointdir_index > 9999:
            raise ValueError("replacement pointdir index exceeds four-digit staging contract")
        records.append({
            **attempt,
            "sample_index": int(sample_index),
            "pointdir_index": int(pointdir_index),
        })
    payload = {
        "schema_version": REPLACEMENT_SAMPLE_SCHEMA_VERSION,
        "context": str(context),
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "sample_xyz": str(sample_path.resolve()),
        "n_candidates": int(len(records)),
        "records": records,
        "point_allocation_manifest": str(allocation_path.resolve()),
        "point_allocation_generation": int(updated["generation"]),
        "point_allocation_sha256": allocation_manifest_sha256(allocation_path),
    }
    atomic_write_json(replacement_sample_manifest_path(round_dir), payload)
    return payload


def _xyz_frame_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0
    count = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            natoms = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError("replacement sample XYZ atom count is invalid") from exc
        if natoms <= 0 or cursor + natoms + 2 > len(lines):
            raise ValueError("replacement sample XYZ frame is truncated")
        cursor += natoms + 2
        count += 1
    return count


def read_replacement_sample(
    round_dir: str | Path,
    *,
    verify_allocation: bool = False,
) -> Dict[str, Any]:
    path = replacement_sample_manifest_path(round_dir)
    if not path.is_file():
        raise FileNotFoundError("replacement sample manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("replacement sample manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise ValueError("replacement sample manifest must be an object")
    if int(data.get("schema_version", -1)) != REPLACEMENT_SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported replacement sample manifest schema")
    context = str(data.get("context") or "")
    if context not in {"bootstrap", "active"}:
        raise ValueError("replacement sample context is invalid")
    iteration = int(data.get("iteration", -1))
    if iteration < 0 or (context == "bootstrap" and iteration != 0):
        raise ValueError("replacement sample iteration is invalid")
    replacement_round = int(data.get("replacement_round", -1))
    if replacement_round <= 0:
        raise ValueError("replacement sample round must be > 0")
    records = data.get("records")
    if (
        not isinstance(records, list)
        or not records
        or len(records) != int(data.get("n_candidates", -1))
    ):
        raise ValueError("replacement sample record count is invalid")
    candidate_ids = set()
    pointdir_indices = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("replacement sample record must be an object")
        candidate_id = str(record.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("replacement sample candidate IDs must be unique")
        candidate_ids.add(candidate_id)
        if int(record.get("sample_index", -1)) != index:
            raise ValueError("replacement sample indexes must be contiguous")
        if int(record.get("round", -1)) != replacement_round:
            raise ValueError("replacement sample record round mismatch")
        if int(record.get("slot_id", -1)) < 0:
            raise ValueError("replacement sample slot_id is invalid")
        if str(record.get("split") or "") not in {"train", "int_val", "ext_val"}:
            raise ValueError("replacement sample split is invalid")
        pointdir_index = int(record.get("pointdir_index", -1))
        if pointdir_index < 0 or pointdir_index in pointdir_indices:
            raise ValueError("replacement sample pointdir indexes must be unique")
        pointdir_indices.add(pointdir_index)
    sample = Path(str(data.get("sample_xyz") or ""))
    if (
        not sample.is_file()
        or sample.is_symlink()
        or sample.parent.resolve() != Path(round_dir).resolve()
    ):
        raise ValueError("replacement sample path is invalid")
    if _xyz_frame_count(sample) != len(records):
        raise ValueError("replacement sample XYZ frame count does not match records")
    allocation_path = Path(str(data.get("point_allocation_manifest") or ""))
    if not allocation_path.is_file() or allocation_path.is_symlink():
        raise ValueError("replacement point-allocation manifest path is invalid")
    allocation_generation = int(data.get("point_allocation_generation", -1))
    if allocation_generation < 0:
        raise ValueError("replacement point-allocation generation is invalid")
    allocation_sha = str(data.get("point_allocation_sha256") or "")
    if (
        len(allocation_sha) != 64
        or any(character not in "0123456789abcdef" for character in allocation_sha)
    ):
        raise ValueError("replacement point-allocation SHA256 is invalid")
    if verify_allocation:
        current = read_point_allocation(allocation_path)
        if int(current["generation"]) != allocation_generation:
            raise ValueError("replacement point-allocation generation has changed")
        if allocation_manifest_sha256(allocation_path) != allocation_sha:
            raise ValueError("replacement point-allocation SHA256 has changed")
    out = dict(data)
    out["sample_xyz"] = str(sample.resolve())
    out["point_allocation_manifest"] = str(allocation_path.resolve())
    return out


__all__ = [
    "REPLACEMENT_SAMPLE_FILENAME",
    "replacement_round_dir",
    "prepare_replacement_round",
    "read_replacement_sample",
]
