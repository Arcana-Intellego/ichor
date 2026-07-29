"""Finite reserve-only replacement sampling for incomplete QM allocations."""
from __future__ import annotations

from .strict_json import StrictJSONDecodeError, strict_json as json
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ichor.core.atoms import Atom, Atoms

from .daemon.state import atomic_write_json, atomic_write_text
from .point_allocation import (
    allocate_replacements,
    allocation_manifest_sha256,
    pending_attempts,
    point_allocation_path,
    read_point_allocation,
)
from .layout import staging_context_dir
from .versioning.manifest import sha256_file


REPLACEMENT_SAMPLE_FILENAME = "REPLACEMENT_SAMPLE.json"
REPLACEMENT_SAMPLE_SCHEMA_VERSION = 2
MAX_MANIFEST_INTEGER = (1 << 63) - 1


def _required_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an exact JSON integer")
    parsed = int(value)
    if parsed < minimum or parsed > MAX_MANIFEST_INTEGER:
        raise ValueError(
            label
            + " must be between "
            + str(minimum)
            + " and "
            + str(MAX_MANIFEST_INTEGER)
        )
    return parsed


def _required_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(label + " must be a lowercase SHA-256 digest")
    return value


def replacement_round_dir(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
) -> Path:
    return (
        staging_context_dir(
            campaign_dir,
            context=str(context),
            iteration=int(iteration),
        )
        / ("replacement_round_" + str(int(replacement_round)).zfill(4))
    )


def replacement_sample_manifest_path(round_dir: str | Path) -> Path:
    return Path(round_dir) / REPLACEMENT_SAMPLE_FILENAME


def _render_xyz(frames: Sequence[Atoms]) -> str:
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
    return "\n".join(lines) + "\n"


def _write_xyz(frames: Sequence[Atoms], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, _render_xyz(frames))


def _active_frame(
    attempt: Mapping[str, Any],
    *,
    expected_result_path: Path | None = None,
    expected_iteration: int | None = None,
    expected_trajectory_sha256: str | None = None,
) -> Atoms:
    result_path = Path(str(attempt.get("result_json") or ""))
    if result_path.is_symlink() or not result_path.is_file():
        raise FileNotFoundError("replacement ARIADNE result is missing: " + str(result_path))
    resolved_result = result_path.resolve()
    if (
        expected_result_path is not None
        and resolved_result != Path(expected_result_path).resolve()
    ):
        raise ValueError("replacement ARIADNE result path is not canonical")
    declared_result_sha = _required_sha256(
        attempt.get("result_sha256"),
        "replacement ARIADNE result_sha256",
    )
    if sha256_file(resolved_result) != declared_result_sha:
        raise ValueError("replacement ARIADNE result SHA-256 mismatch")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except StrictJSONDecodeError as exc:
        if "non-standard JSON constant" in str(exc):
            raise ValueError(
                "replacement ARIADNE result contains non-finite coordinates "
                "or diagnostics: " + str(result_path)
            ) from exc
        raise ValueError(
            "replacement ARIADNE result is unreadable: " + str(result_path)
        ) from exc
    except (OSError, ValueError) as exc:
        raise ValueError("replacement ARIADNE result is unreadable: " + str(result_path)) from exc
    landing_safety = result.get("landing_safety")
    if not isinstance(landing_safety, dict) or landing_safety.get("accepted") is not True:
        raise ValueError("replacement ARIADNE landing is not explicitly safe")
    attempt_safety = attempt.get("landing_safety")
    if not isinstance(attempt_safety, dict) or attempt_safety.get("accepted") is not True:
        raise ValueError("replacement allocation lacks accepted landing-safety evidence")
    if expected_iteration is not None:
        from .handoff_manifests import validate_ariadne_result

        validate_ariadne_result(
            result,
            expected_iteration=_required_integer(
                expected_iteration,
                "replacement ARIADNE expected iteration",
                minimum=1,
            ),
            seed_record={
                "seed_id": _required_integer(
                    attempt.get("seed_id"),
                    "replacement ARIADNE seed_id",
                    minimum=1,
                ),
                "seed_uid": str(attempt.get("seed_uid") or ""),
                "frame_id": attempt.get("seed_frame_id"),
            },
            expected_trajectory_sha256=expected_trajectory_sha256,
        )
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


def _active_frames(
    campaign_dir: Path,
    *,
    iteration: int,
    attempts: Sequence[Mapping[str, Any]],
) -> List[Atoms]:
    from .handoff_manifests import authoritative_ariadne_candidate_frames

    resolved_iteration = _required_integer(
        iteration,
        "replacement active iteration",
        minimum=1,
    )
    manifest, _frames, accepted_records = authoritative_ariadne_candidate_frames(
        campaign_dir,
        resolved_iteration,
    )
    accepted_by_uid = {
        str(record.get("seed_uid") or ""): record for record in accepted_records
    }
    if len(accepted_by_uid) != len(accepted_records):
        raise ValueError("ARIADNE accepted records contain duplicate seed identities")
    selected: List[Atoms] = []
    for attempt in attempts:
        seed_uid = str(attempt.get("seed_uid") or "")
        source = accepted_by_uid.get(seed_uid)
        if source is None:
            raise ValueError(
                "replacement candidate is not present in accepted ARIADNE results"
            )
        for field in ("seed_id", "result_sha256"):
            if attempt.get(field) != source.get(field):
                raise ValueError(
                    "replacement candidate " + field + " disagrees with ARIADNE results"
                )
        if Path(str(attempt.get("result_json") or "")).resolve() != Path(
            str(source.get("result_json") or "")
        ).resolve():
            raise ValueError(
                "replacement candidate result path disagrees with ARIADNE results"
            )
        selected.append(
            _active_frame(
                attempt,
                expected_result_path=Path(str(source["result_json"])),
                expected_iteration=resolved_iteration,
                expected_trajectory_sha256=(
                    str(manifest.get("trajectory_sha256") or "") or None
                ),
            )
        )
    return selected


def _bootstrap_frames(campaign_dir: Path, attempts: Sequence[Mapping[str, Any]]) -> List[Atoms]:
    from .acquisition.trajectory_pool import TrajectoryPool

    pool = TrajectoryPool.load(campaign_dir)
    frames = pool.to_atoms_list()
    selected: List[Atoms] = []
    for attempt in attempts:
        declared_pool_sha = _required_sha256(
            attempt.get("pool_sha256"),
            "bootstrap replacement pool_sha256",
        )
        if declared_pool_sha != pool.sha256:
            raise ValueError(
                "bootstrap replacement candidate belongs to a different trajectory pool"
            )
        frame_id = attempt.get("frame_id")
        index = _required_integer(
            frame_id,
            "bootstrap replacement frame_id",
        )
        if index < 0 or index >= len(frames):
            raise ValueError("bootstrap replacement frame_id is outside the pool")
        selected.append(frames[index].copy())
    return selected


def _replacement_material(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    allow_allocate: bool,
    expected_campaign_uid: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any], List[Dict[str, Any]], List[Atoms]]:
    campaign = Path(campaign_dir)
    allocation_path = point_allocation_path(
        campaign,
        context=context,
        iteration=int(iteration),
    )
    current = read_point_allocation(
        allocation_path,
        expected_campaign_uid=expected_campaign_uid,
        expected_context=str(context),
        expected_iteration=int(iteration),
    )
    if (current.get("summary") or {}).get("complete") is True:
        raise ValueError("point allocation is already complete")
    existing_pending = pending_attempts(current)
    if existing_pending:
        pending_rounds = {
            _required_integer(
                attempt.get("round"),
                "pending replacement round",
                minimum=1,
            )
            for attempt in existing_pending
        }
        if pending_rounds != {int(replacement_round)}:
            raise ValueError(
                "point allocation already has pending candidates for replacement round(s) "
                + repr(sorted(pending_rounds))
            )
        updated = current
    else:
        if not bool(allow_allocate):
            raise ValueError(
                "replacement sample recovery requires an existing pending round"
            )
        updated = allocate_replacements(
            allocation_path,
            replacement_round=int(replacement_round),
            expected_generation=_required_integer(
                current.get("generation"),
                "point-allocation generation",
            ),
        )
    attempts = [
        attempt
        for attempt in pending_attempts(updated)
        if _required_integer(
            attempt.get("round"), "replacement attempt round", minimum=1
        )
        == int(replacement_round)
    ]
    if not attempts:
        raise ValueError("replacement round allocated no candidates")
    if str(context) == "bootstrap":
        frames = _bootstrap_frames(campaign, attempts)
    elif str(context) == "active":
        frames = _active_frames(
            campaign,
            iteration=int(iteration),
            attempts=attempts,
        )
    else:
        raise ValueError("replacement context must be bootstrap or active")
    return allocation_path, updated, [dict(attempt) for attempt in attempts], frames


def _replacement_payload(
    *,
    allocation_path: Path,
    allocation: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
    sample_text: str,
    context: str,
    iteration: int,
    replacement_round: int,
) -> Dict[str, Any]:
    records = []
    target_total = _required_integer(
        allocation["targets"]["total"], "point-allocation target total"
    )
    for sample_index, attempt in enumerate(attempts):
        reserve_rank = _required_integer(
            attempt.get("reserve_rank"),
            "replacement reserve_rank",
        )
        pointdir_index = target_total + reserve_rank
        records.append(
            {
                **dict(attempt),
                "sample_index": int(sample_index),
                "pointdir_index": int(pointdir_index),
            }
        )
    sample_bytes = sample_text.encode("utf-8")
    return {
        "schema_version": REPLACEMENT_SAMPLE_SCHEMA_VERSION,
        "context": str(context),
        "iteration": int(iteration),
        "replacement_round": int(replacement_round),
        "sample_xyz": {
            "path": "replacement-SAMPLE.xyz",
            "size": int(len(sample_bytes)),
            "sha256": hashlib.sha256(sample_bytes).hexdigest(),
        },
        "n_candidates": int(len(records)),
        "records": records,
        "point_allocation_manifest": str(allocation_path.resolve()),
        "point_allocation_generation": _required_integer(
            allocation["generation"], "point-allocation generation"
        ),
        "point_allocation_sha256": allocation_manifest_sha256(allocation_path),
    }


def prepare_replacement_round(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    campaign = Path(campaign_dir)
    allocation_path, updated, attempts, frames = _replacement_material(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        allow_allocate=True,
        expected_campaign_uid=expected_campaign_uid,
    )

    round_dir = replacement_round_dir(
        campaign,
        context=context,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    round_dir.mkdir(parents=True, exist_ok=True)
    sample_path = round_dir / "replacement-SAMPLE.xyz"
    sample_text = _render_xyz(frames)
    atomic_write_text(sample_path, sample_text)
    payload = _replacement_payload(
        allocation_path=allocation_path,
        allocation=updated,
        attempts=attempts,
        sample_text=sample_text,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    atomic_write_json(replacement_sample_manifest_path(round_dir), payload)
    return payload


def _xyz_frames(path: Path) -> List[Atoms]:
    from ichor.core.files.xyz.strict_xyz import read_xyz_frames

    frames = read_xyz_frames(path)
    if not frames:
        raise ValueError("replacement sample XYZ is empty")
    expected_atom_types = tuple(atom.type for atom in frames[0])
    for frame in frames[1:]:
        if tuple(atom.type for atom in frame) != expected_atom_types:
            raise ValueError(
                "replacement sample XYZ changes atom identity or order"
            )
    return frames


def read_replacement_sample(
    round_dir: str | Path,
    *,
    verify_allocation: bool = False,
    expected_campaign_uid: Optional[str] = None,
    expected_context: Optional[str] = None,
    expected_iteration: Optional[int] = None,
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
    if _required_integer(
        data.get("schema_version"), "replacement sample schema_version"
    ) != REPLACEMENT_SAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported replacement sample manifest schema")
    context = data.get("context")
    if not isinstance(context, str):
        raise ValueError("replacement sample context must be a string")
    if context not in {"bootstrap", "active"}:
        raise ValueError("replacement sample context is invalid")
    iteration = _required_integer(data.get("iteration"), "replacement sample iteration")
    if iteration < 0 or (context == "bootstrap" and iteration != 0):
        raise ValueError("replacement sample iteration is invalid")
    replacement_round = _required_integer(
        data.get("replacement_round"),
        "replacement sample round",
        minimum=1,
    )
    records = data.get("records")
    if (
        not isinstance(records, list)
        or not records
        or len(records)
        != _required_integer(
            data.get("n_candidates"), "replacement sample n_candidates", minimum=1
        )
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
        if _required_integer(
            record.get("sample_index"), "replacement sample_index"
        ) != index:
            raise ValueError("replacement sample indexes must be contiguous")
        if _required_integer(
            record.get("round"), "replacement record round", minimum=1
        ) != replacement_round:
            raise ValueError("replacement sample record round mismatch")
        _required_integer(record.get("slot_id"), "replacement sample slot_id")
        if str(record.get("split") or "") not in {"train", "int_val", "ext_val"}:
            raise ValueError("replacement sample split is invalid")
        pointdir_index = _required_integer(
            record.get("pointdir_index"), "replacement sample pointdir_index"
        )
        if pointdir_index < 0 or pointdir_index in pointdir_indices:
            raise ValueError("replacement sample pointdir indexes must be unique")
        pointdir_indices.add(pointdir_index)
    sample_binding = data.get("sample_xyz")
    if not isinstance(sample_binding, dict) or set(sample_binding) != {
        "path",
        "size",
        "sha256",
    }:
        raise ValueError("replacement sample XYZ binding is invalid")
    if sample_binding.get("path") != "replacement-SAMPLE.xyz":
        raise ValueError("replacement sample XYZ path is noncanonical")
    sample = Path(round_dir) / "replacement-SAMPLE.xyz"
    if (
        not sample.is_file()
        or sample.is_symlink()
        or sample.parent.resolve() != Path(round_dir).resolve()
    ):
        raise ValueError("replacement sample path is invalid")
    if _required_integer(
        sample_binding.get("size"), "replacement sample XYZ size"
    ) != int(sample.stat().st_size):
        raise ValueError("replacement sample XYZ size mismatch")
    if _required_sha256(
        sample_binding.get("sha256"), "replacement sample XYZ SHA-256"
    ) != sha256_file(sample):
        raise ValueError("replacement sample XYZ SHA-256 mismatch")
    if len(_xyz_frames(sample)) != len(records):
        raise ValueError("replacement sample XYZ frame count does not match records")
    allocation_path = Path(str(data.get("point_allocation_manifest") or ""))
    if not allocation_path.is_file() or allocation_path.is_symlink():
        raise ValueError("replacement point-allocation manifest path is invalid")
    allocation_generation = _required_integer(
        data.get("point_allocation_generation"),
        "replacement point-allocation generation",
    )
    allocation_sha = _required_sha256(
        data.get("point_allocation_sha256"),
        "replacement point-allocation SHA-256",
    )
    if verify_allocation:
        current = read_point_allocation(
            allocation_path,
            expected_campaign_uid=expected_campaign_uid,
            expected_context=expected_context,
            expected_iteration=expected_iteration,
        )
        if int(current["generation"]) != allocation_generation:
            raise ValueError("replacement point-allocation generation has changed")
        if allocation_manifest_sha256(allocation_path) != allocation_sha:
            raise ValueError("replacement point-allocation SHA256 has changed")
    out = dict(data)
    out["sample_xyz_binding"] = dict(sample_binding)
    out["sample_xyz"] = str(sample.resolve())
    out["point_allocation_manifest"] = str(allocation_path.resolve())
    return out


def read_replacement_sample_strict(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a replacement sample and join it exactly to current allocation."""
    campaign = Path(campaign_dir)
    canonical_round_dir = replacement_round_dir(
        campaign,
        context=context,
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    data = read_replacement_sample(
        canonical_round_dir,
        verify_allocation=True,
        expected_campaign_uid=expected_campaign_uid,
        expected_context=str(context),
        expected_iteration=int(iteration),
    )
    if str(data.get("context")) != str(context):
        raise ValueError("replacement sample context does not match its campaign phase")
    if int(data.get("iteration", -1)) != int(iteration):
        raise ValueError("replacement sample iteration does not match its campaign phase")
    if int(data.get("replacement_round", -1)) != int(replacement_round):
        raise ValueError("replacement sample round does not match its campaign phase")
    allocation_path = point_allocation_path(
        campaign,
        context=context,
        iteration=int(iteration),
    ).resolve(strict=False)
    if Path(str(data["point_allocation_manifest"])).resolve(strict=False) != allocation_path:
        raise ValueError("replacement sample references a non-canonical allocation manifest")
    allocation = read_point_allocation(
        allocation_path,
        expected_campaign_uid=expected_campaign_uid,
        expected_context=str(context),
        expected_iteration=int(iteration),
    )
    expected_attempts = [
        attempt
        for attempt in pending_attempts(allocation)
        if int(attempt.get("round", -1)) == int(replacement_round)
    ]
    records = list(data.get("records") or [])
    if len(records) != len(expected_attempts) or not records:
        raise ValueError(
            "replacement sample does not cover exactly the current pending attempts"
        )
    target_total = int(allocation["targets"]["total"])
    for sample_index, (record, expected) in enumerate(
        zip(records, expected_attempts)
    ):
        core = {
            key: value
            for key, value in dict(record).items()
            if key not in {"sample_index", "pointdir_index"}
        }
        if core != expected:
            raise ValueError(
                "replacement sample record does not match allocation attempt at index "
                + str(sample_index)
            )
        expected_pointdir_index = target_total + int(expected["reserve_rank"])
        if int(record.get("pointdir_index", -1)) != expected_pointdir_index:
            raise ValueError(
                "replacement sample pointdir index does not match reserve rank"
            )
    expected_sample = (canonical_round_dir / "replacement-SAMPLE.xyz").resolve(
        strict=False
    )
    if Path(str(data["sample_xyz"])).resolve(strict=False) != expected_sample:
        raise ValueError("replacement sample XYZ is not the canonical round artefact")
    return data


def _replacement_sample_repair_plan(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify derived replacement evidence against pending allocation."""
    campaign = Path(campaign_dir)
    round_dir = replacement_round_dir(
        campaign,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
    )
    sample_path = round_dir / "replacement-SAMPLE.xyz"
    manifest_path = replacement_sample_manifest_path(round_dir)
    if sample_path.is_symlink() or manifest_path.is_symlink():
        return {
            "state": "conflicting",
            "reason": "replacement sample evidence must not be symlinked",
            "round_dir": round_dir,
            "sample_path": sample_path,
            "manifest_path": manifest_path,
        }
    try:
        loaded = read_replacement_sample_strict(
            campaign,
            context=str(context),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            expected_campaign_uid=expected_campaign_uid,
        )
    except (FileNotFoundError, OSError, ValueError) as strict_error:
        sample_present = sample_path.exists()
        manifest_present = manifest_path.exists()
        if sample_present and manifest_present:
            return {
                "state": "conflicting",
                "reason": str(strict_error),
                "round_dir": round_dir,
                "sample_path": sample_path,
                "manifest_path": manifest_path,
            }
    else:
        return {
            "state": "valid",
            "reason": None,
            "round_dir": round_dir,
            "sample_path": sample_path,
            "manifest_path": manifest_path,
            "n_candidates": int(loaded["n_candidates"]),
            "loaded": loaded,
        }

    try:
        allocation_path, allocation, attempts, frames = _replacement_material(
            campaign,
            context=str(context),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
            allow_allocate=False,
            expected_campaign_uid=expected_campaign_uid,
        )
        sample_text = _render_xyz(frames)
        expected_manifest = _replacement_payload(
            allocation_path=allocation_path,
            allocation=allocation,
            attempts=attempts,
            sample_text=sample_text,
            context=str(context),
            iteration=int(iteration),
            replacement_round=int(replacement_round),
        )
        if sample_present:
            if not sample_path.is_file() or sample_path.read_text(
                encoding="utf-8"
            ) != sample_text:
                raise ValueError(
                    "existing replacement sample conflicts with pending allocation"
                )
        if manifest_present:
            try:
                existing_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as exc:
                raise ValueError(
                    "existing replacement sample manifest is unreadable"
                ) from exc
            if existing_manifest != expected_manifest:
                raise ValueError(
                    "existing replacement sample manifest conflicts with pending allocation"
                )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {
            "state": "conflicting",
            "reason": str(exc),
            "round_dir": round_dir,
            "sample_path": sample_path,
            "manifest_path": manifest_path,
        }

    return {
        "state": (
            "missing_rebuildable"
            if not sample_present and not manifest_present
            else "partial_rebuildable"
        ),
        "reason": None,
        "round_dir": round_dir,
        "sample_path": sample_path,
        "manifest_path": manifest_path,
        "sample_text": sample_text,
        "expected_manifest": expected_manifest,
        "n_candidates": int(len(attempts)),
    }


def inspect_replacement_sample_recovery(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a read-only classification of replacement-sample evidence."""
    plan = _replacement_sample_repair_plan(
        campaign_dir,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )
    return {
        key: (
            str(value)
            if key in {"round_dir", "sample_path", "manifest_path"}
            else value
        )
        for key, value in plan.items()
        if key not in {"loaded", "sample_text", "expected_manifest"}
    }


def ensure_replacement_sample_strict(
    campaign_dir: str | Path,
    *,
    context: str,
    iteration: int,
    replacement_round: int,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """Recreate only missing derived sample evidence from pending allocation."""
    plan = _replacement_sample_repair_plan(
        campaign_dir,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )
    state = str(plan["state"])
    if state == "valid":
        return dict(plan["loaded"])
    if state not in {"missing_rebuildable", "partial_rebuildable"}:
        raise ValueError(
            "replacement sample evidence conflicts with pending allocation: "
            + str(plan.get("reason") or "unknown conflict")
        )
    round_dir = Path(plan["round_dir"])
    sample_path = Path(plan["sample_path"])
    manifest_path = Path(plan["manifest_path"])
    round_dir.mkdir(parents=True, exist_ok=True)
    if not sample_path.exists():
        atomic_write_text(sample_path, str(plan["sample_text"]))
    if not manifest_path.exists():
        atomic_write_json(manifest_path, dict(plan["expected_manifest"]))
    return read_replacement_sample_strict(
        campaign_dir,
        context=str(context),
        iteration=int(iteration),
        replacement_round=int(replacement_round),
        expected_campaign_uid=expected_campaign_uid,
    )


__all__ = [
    "REPLACEMENT_SAMPLE_FILENAME",
    "replacement_round_dir",
    "ensure_replacement_sample_strict",
    "inspect_replacement_sample_recovery",
    "prepare_replacement_round",
    "read_replacement_sample",
    "read_replacement_sample_strict",
]
