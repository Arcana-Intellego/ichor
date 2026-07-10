"""Per-pointdir provenance ledger.

Every committed `.pointdir` in `QM_REFERENCE_DATA/iteration-NNNNNN/` carries a
`.provenance.json` sidecar that traces the point back through the
adversarial-attack pipeline: which MD frame seeded it, which 50 neighbour
frames built its local subspace, what ARIADNE did to it, whether the
post-descent geometry passed the anti-overlap check, and where Phase-B
FPS placed it in the diversity ranking.

In parallel, an append-only flat index lives at

    <campaign>/.DATA/ACTIVE_LEARNING/seed_frame_id_index.json

so the daemon's SEED_SELECT phase can compute "frames already seeded into
the QM reference data" in O(1) reads rather than O(n) sidecar scans. The fast
index is updated incrementally on every APPEND (a few records per
iteration), atomically via tempfile + os.replace.

every sidecar JSON pins back to the campaign uid + iteration, so a sweep
across the QM reference data can always trace each committed pointdir back to
the seed and the MD frame it descended from. the flat index file off to
the side is the cheap version of the same lookup -- a one-pass read at
SEED_SELECT time so the daemon can compute which frames are already
forbidden without walking thousands of sidecars over NFS.

Writes are atomic: `daemon.state.atomic_write_json` is the canonical
serialiser used here (tempfile + fsync + os.replace + parent-dir fsync).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from contextlib import contextmanager

from ..daemon.state import atomic_write_json


# the read-modify-write helpers below need an actual lock, not just the
# atomic-rename trick atomic_write_json gives us. two enrich callers
# racing on the same .provenance.json would both read the old file, each
# would compute its own update, and whichever wrote last would silently
# bin the other's work. wrapping the load->modify->store body in a
# portalocker flock closes that window. portalocker is already pulled in
# elsewhere for the campaign-wide lock so no new dep here.

_PROVENANCE_LOCK_FILENAME = ".provenance.lock"
_INDEX_LOCK_FILENAME = "seed_frame_id_index.lock"
_RECENT_SEEDS_LOCK_FILENAME = "recent_seeds.lock"
_DEFAULT_LOCK_TIMEOUT_SECONDS = 30.0


@contextmanager
def _pointdir_lock(pointdir, timeout: float = _DEFAULT_LOCK_TIMEOUT_SECONDS):
    """Exclusive lock for read-modify-write of a single pointdir's
    .provenance.json. Lockfile lives inside the pointdir."""
    import portalocker
    pointdir = Path(pointdir)
    pointdir.mkdir(parents=True, exist_ok=True)
    lock_path = pointdir / _PROVENANCE_LOCK_FILENAME
    with portalocker.Lock(
        str(lock_path), mode="a", flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        timeout=timeout,
    ) as f:
        yield f


@contextmanager
def _index_lock(campaign_dir, timeout: float = _DEFAULT_LOCK_TIMEOUT_SECONDS):
    """Exclusive lock for read-modify-write of seed_frame_id_index.json."""
    import portalocker
    campaign_dir = Path(campaign_dir)
    lock_dir = campaign_dir / ".DATA" / "ACTIVE_LEARNING"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / _INDEX_LOCK_FILENAME
    with portalocker.Lock(
        str(lock_path), mode="a", flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        timeout=timeout,
    ) as f:
        yield f


@contextmanager
def _recent_seeds_lock(campaign_dir, timeout: float = _DEFAULT_LOCK_TIMEOUT_SECONDS):
    """Exclusive lock for read-modify-write of recent_seeds.json."""
    import portalocker
    campaign_dir = Path(campaign_dir)
    lock_dir = campaign_dir / ".DATA" / "ACTIVE_LEARNING"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / _RECENT_SEEDS_LOCK_FILENAME
    with portalocker.Lock(
        str(lock_path), mode="a", flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
        timeout=timeout,
    ) as f:
        yield f


__all__ = [
    "PROVENANCE_FILENAME",
    "PROVENANCE_SCHEMA_VERSION",
    "SEED_FRAME_ID_INDEX_FILENAME",
    "INDEX_SCHEMA_VERSION",
    "RECENT_SEEDS_FILENAME",
    "RECENT_SEEDS_SCHEMA_VERSION",
    "DEFAULT_RECENT_SEEDS_COOLDOWN",
    "ProvenanceError",
    "write_seed_provenance",
    "enrich_with_ariadne",
    "enrich_with_phase_b",
    "enrich_with_point_allocation",
    "enrich_with_anti_overlap",
    "enrich_with_error_calibration_input",
    "read_provenance",
    "validate_provenance",
    "ensure_index",
    "append_to_index",
    "load_index",
    "load_training_seed_frame_ids",
    "repair_index_from_committed_pointdirs",
    "seed_frame_ids_from_committed_pointdirs",
    "append_recent_seeds",
    "load_recent_seeds_payload",
    "load_recent_seed_frame_ids",
]


PROVENANCE_FILENAME = ".provenance.json"
PROVENANCE_SCHEMA_VERSION = 2

# Index lives under <campaign>/.DATA/ACTIVE_LEARNING/
SEED_FRAME_ID_INDEX_FILENAME = "seed_frame_id_index.json"
INDEX_SCHEMA_VERSION = 1


class ProvenanceError(RuntimeError):
    """Raised on malformed provenance JSON or index files."""


# ---------------------------------------------------------------------------
# Per-pointdir sidecar

def _provenance_path(pointdir: Union[str, Path]) -> Path:
    return Path(pointdir) / PROVENANCE_FILENAME


def write_seed_provenance(
    pointdir: Union[str, Path],
    *,
    campaign_uid: str,
    iteration: int,
    trajectory_sha256: str,
    seed_frame_id: Optional[int],
    seed_selection_origin: str,
    seed_variance_at_selection: Optional[float],
    subspace_neighbour_frame_ids: Sequence[int],
    subspace_dimension: int,
    subspace_eigenvalues: Sequence[float],
    mode_weighting_policy: str = "variance",
) -> Path:
    """Write the initial provenance sidecar at SEED_SELECT time.

    The ARIADNE / anti_overlap / phase_b blocks are added later by the
    `enrich_with_*` helpers; this call seeds the schema with the campaign
    + seed + subspace metadata that's known immediately.
    """
    payload: Dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "iteration": int(iteration),
        "trajectory_sha256": str(trajectory_sha256),
        "seed": {
            "frame_id": (
                int(seed_frame_id) if seed_frame_id is not None else None
            ),
            "selection_origin": str(seed_selection_origin),
            "variance_at_selection": (
                float(seed_variance_at_selection)
                if seed_variance_at_selection is not None
                else None
            ),
        },
        "subspace": {
            "neighbour_frame_ids": [int(i) for i in subspace_neighbour_frame_ids],
            "dimension": int(subspace_dimension),
            "eigenvalues": [float(v) for v in subspace_eigenvalues],
            "mode_weighting_policy": str(mode_weighting_policy),
        },
        "ariadne": None,
        "anti_overlap": None,
        "error_calibration_input": None,
        "phase_b": None,
        "point_allocation": None,
    }
    p = _provenance_path(pointdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, payload)
    return p


def read_provenance(pointdir: Union[str, Path]) -> Dict[str, Any]:
    """Read and validate the provenance JSON for one pointdir."""
    p = _provenance_path(pointdir)
    if not p.is_file():
        raise FileNotFoundError("no provenance at " + str(p))
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ProvenanceError(str(p) + ": must be a JSON object")
    schema = int(data.get("schema_version", -1))
    if schema != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError(
            str(p) + ": schema_version " + str(schema)
            + " != " + str(PROVENANCE_SCHEMA_VERSION)
        )
    return data


def _expect_int_or_none(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProvenanceError(label + " must be an integer or null")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(label + " must be an integer or null") from exc


def validate_provenance(
    pointdir: Union[str, Path],
    *,
    campaign_uid: Optional[str] = None,
    iteration: Optional[int] = None,
    trajectory_sha256: Optional[str] = None,
    seed_frame_id: Optional[int] = None,
    require_phase_b_selected: Optional[bool] = None,
    allocation_split: Optional[str] = None,
    allocation_slot_id: Optional[int] = None,
    allocation_candidate_id: Optional[str] = None,
    allocation_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Read provenance and validate the campaign/seed handoff contract."""
    data = read_provenance(pointdir)
    if not isinstance(data.get("campaign_uid"), str) or not data.get("campaign_uid"):
        raise ProvenanceError("provenance campaign_uid is missing or invalid")
    if campaign_uid is not None and str(data.get("campaign_uid")) != str(campaign_uid):
        raise ProvenanceError("provenance campaign_uid mismatch")
    try:
        prov_iteration = int(data.get("iteration"))
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("provenance iteration must be an integer") from exc
    if prov_iteration < 0:
        raise ProvenanceError("provenance iteration must be >= 0")
    if iteration is not None and prov_iteration != int(iteration):
        raise ProvenanceError("provenance iteration mismatch")
    if not isinstance(data.get("trajectory_sha256"), str):
        raise ProvenanceError("provenance trajectory_sha256 is missing or invalid")
    if (
        trajectory_sha256 is not None
        and str(data.get("trajectory_sha256")) != str(trajectory_sha256)
    ):
        raise ProvenanceError("provenance trajectory_sha256 mismatch")

    seed = data.get("seed")
    if not isinstance(seed, dict):
        raise ProvenanceError("provenance seed block must be an object")
    frame_id = _expect_int_or_none(seed.get("frame_id"), "provenance seed.frame_id")
    if seed_frame_id is not None and frame_id != int(seed_frame_id):
        raise ProvenanceError("provenance seed.frame_id mismatch")
    if "selection_origin" in seed and not isinstance(seed.get("selection_origin"), str):
        raise ProvenanceError("provenance seed.selection_origin must be a string")

    subspace = data.get("subspace")
    if not isinstance(subspace, dict):
        raise ProvenanceError("provenance subspace block must be an object")
    try:
        dimension = int(subspace.get("dimension"))
    except (TypeError, ValueError) as exc:
        raise ProvenanceError("provenance subspace.dimension must be an integer") from exc
    if dimension < 0:
        raise ProvenanceError("provenance subspace.dimension must be >= 0")
    neighbours = subspace.get("neighbour_frame_ids")
    if not isinstance(neighbours, list):
        raise ProvenanceError("provenance subspace.neighbour_frame_ids must be a list")
    for value in neighbours:
        _expect_int_or_none(value, "provenance subspace.neighbour_frame_ids")

    phase_b = data.get("phase_b")
    if require_phase_b_selected is not None:
        if not isinstance(phase_b, dict):
            raise ProvenanceError("provenance phase_b block must be an object")
        selected = bool(phase_b.get("selected_after_fps", False))
        if selected != bool(require_phase_b_selected):
            raise ProvenanceError("provenance phase_b.selected_after_fps mismatch")
    allocation = data.get("point_allocation")
    if (
        allocation_split is not None
        or allocation_slot_id is not None
        or allocation_candidate_id is not None
        or allocation_context is not None
    ):
        if not isinstance(allocation, dict):
            raise ProvenanceError("provenance point_allocation block must be an object")
        split = str(allocation.get("split") or "")
        if split not in {"train", "int_val", "ext_val"}:
            raise ProvenanceError("provenance point_allocation.split is invalid")
        if allocation_split is not None and split != str(allocation_split):
            raise ProvenanceError("provenance point_allocation.split mismatch")
        slot_id = _expect_int_or_none(
            allocation.get("slot_id"),
            "provenance point_allocation.slot_id",
        )
        if slot_id is None or slot_id < 0:
            raise ProvenanceError("provenance point_allocation.slot_id must be >= 0")
        if allocation_slot_id is not None and slot_id != int(allocation_slot_id):
            raise ProvenanceError("provenance point_allocation.slot_id mismatch")
        candidate_id = allocation.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ProvenanceError(
                "provenance point_allocation.candidate_id is invalid"
            )
        if (
            allocation_candidate_id is not None
            and candidate_id != str(allocation_candidate_id)
        ):
            raise ProvenanceError(
                "provenance point_allocation.candidate_id mismatch"
            )
        context = allocation.get("context")
        if context not in {"bootstrap", "active"}:
            raise ProvenanceError("provenance point_allocation.context is invalid")
        if allocation_context is not None and context != str(allocation_context):
            raise ProvenanceError("provenance point_allocation.context mismatch")
    return data


def _merge_section(pointdir: Union[str, Path], section_name: str, section_payload: Dict[str, Any]) -> Path:
    """workhorse behind every enrich_with_* helper. atomically rewrites
    the sidecar with the named section replaced; every other section
    is preserved as-is.

    the load-modify-store cycle sits inside a per-pointdir flock --
    two enrich calls hitting the same sidecar at the same time (say the
    daemon mid-postprocess plus an out-of-band reconcile worker) would
    otherwise read the same starting bytes and the second writer's
    changes would silently win. atomic_write_json only guarantees you
    won't see a half-written file; it doesn't guarantee you won't see
    a stale one.
    """
    p = _provenance_path(pointdir)
    if not p.is_file():
        raise FileNotFoundError(
            "no provenance to enrich at " + str(p)
            + "; write_seed_provenance must run first"
        )
    with _pointdir_lock(pointdir):
        data = read_provenance(pointdir)
        data[section_name] = dict(section_payload)
        atomic_write_json(p, data)
    return p


def enrich_with_ariadne(
    pointdir: Union[str, Path],
    *,
    alpha_initial: float,
    alpha_final: float,
    n_evaluations: int,
    fell_back_to_ds: bool,
    wall_seconds: float,
    return_code: Optional[int] = None,
) -> Path:
    """Append the ARIADNE-run record to an existing provenance sidecar."""
    payload: Dict[str, Any] = {
        "alpha_initial": float(alpha_initial),
        "alpha_final": float(alpha_final),
        "n_evaluations": int(n_evaluations),
        "fell_back_to_ds": bool(fell_back_to_ds),
        "wall_seconds": float(wall_seconds),
    }
    if return_code is not None:
        payload["return_code"] = int(return_code)
    return _merge_section(pointdir, "ariadne", payload)


def enrich_with_anti_overlap(
    pointdir: Union[str, Path],
    *,
    min_whitened_distance_to_training: Optional[float],
    passed: bool,
    flag: Optional[str] = None,
) -> Path:
    """Append the post-ARIADNE anti-overlap result. `flag` is one of
    ``"moved_too_little"`` / ``"moved_too_far"`` / ``None``."""
    payload: Dict[str, Any] = {
        "min_whitened_distance_to_training": (
            float(min_whitened_distance_to_training)
            if min_whitened_distance_to_training is not None
            else None
        ),
        "passed": bool(passed),
        "flag": (None if flag is None else str(flag)),
    }
    return _merge_section(pointdir, "anti_overlap", payload)


def enrich_with_error_calibration_input(
    pointdir: Union[str, Path],
    payload: Dict[str, Any],
) -> Path:
    """Append selected-landing prediction diagnostics for later calibration."""
    return _merge_section(pointdir, "error_calibration_input", dict(payload))


def enrich_with_phase_b(
    pointdir: Union[str, Path],
    *,
    selected_after_fps: bool,
    diversity_rank: Optional[int],
    descriptor_used: str,
    candidate_id: Optional[str] = None,
    reserve_candidate: bool = False,
) -> Path:
    """Append the Phase-B (post-FPS) record."""
    payload: Dict[str, Any] = {
        "selected_after_fps": bool(selected_after_fps),
        "diversity_rank": (None if diversity_rank is None else int(diversity_rank)),
        "descriptor_used": str(descriptor_used),
        "reserve_candidate": bool(reserve_candidate),
    }
    if candidate_id is not None:
        payload["candidate_id"] = str(candidate_id)
    return _merge_section(pointdir, "phase_b", payload)


def enrich_with_point_allocation(
    pointdir: Union[str, Path],
    *,
    candidate_id: str,
    context: str,
    slot_id: int,
    split: str,
    replacement_round: int = 0,
    allocation_manifest_sha256: Optional[str] = None,
) -> Path:
    """Attach the pre-QM allocation slot consumed by this candidate."""
    split_name = str(split)
    if split_name not in {"train", "int_val", "ext_val"}:
        raise ValueError("point-allocation split is invalid: " + repr(split))
    payload: Dict[str, Any] = {
        "candidate_id": str(candidate_id),
        "context": str(context),
        "slot_id": int(slot_id),
        "split": split_name,
        "replacement_round": int(replacement_round),
    }
    if allocation_manifest_sha256 is not None:
        payload["allocation_manifest_sha256"] = str(allocation_manifest_sha256)
    return _merge_section(pointdir, "point_allocation", payload)


# ---------------------------------------------------------------------------
# Flat index file -- O(1) read at SEED_SELECT time

def _index_path(campaign_dir: Union[str, Path]) -> Path:
    return Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / SEED_FRAME_ID_INDEX_FILENAME


def _empty_index_payload() -> Dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "records": [],
    }


def ensure_index(campaign_dir: Union[str, Path]) -> Path:
    """Create an empty index file if absent. No-op if it already exists."""
    p = _index_path(campaign_dir)
    if p.is_file():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, _empty_index_payload())
    return p


def load_index(campaign_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read the index payload. Raises ProvenanceError on schema drift.

    Treats a missing file as an empty index (callers can therefore use this
    cheaply without first calling :func:`ensure_index`).
    """
    p = _index_path(campaign_dir)
    if not p.is_file():
        return _empty_index_payload()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            "seed_frame_id_index.json failed to parse: " + str(exc)
        ) from exc
    if not isinstance(data, dict):
        raise ProvenanceError("seed_frame_id_index.json must be a JSON object")
    schema = int(data.get("schema_version", -1))
    if schema != INDEX_SCHEMA_VERSION:
        raise ProvenanceError(
            "seed_frame_id_index.json schema_version " + str(schema)
            + " != " + str(INDEX_SCHEMA_VERSION)
        )
    records = data.get("records")
    if not isinstance(records, list):
        raise ProvenanceError("seed_frame_id_index.json `records` must be a list")
    return data


def append_to_index(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
    pointdir_name: str,
    seed_frame_id: Optional[int],
) -> Path:
    """add one (iteration, pointdir_name, seed_frame_id) record to the
    flat index file. the whole add is atomic.

    each call reads the whole list, appends to it, writes it back. that
    sounds wasteful but the file is small -- at iteration 30 with 50
    seeds per iteration we're under 1500 records, well below 100 KB,
    and the write cost is dwarfed by the actual quantum-chemistry work
    happening alongside it. the alternative would be scanning the per-
    pointdir sidecars at SEED_SELECT time instead -- thousands of NFS
    reads per iteration, much worse.
    """
    # flock around the load -> append -> write cycle so two APPEND
    # workers can't lose records to a clobber race.
    with _index_lock(campaign_dir):
        data = load_index(campaign_dir)
        record = {
            "iteration": int(iteration),
            "pointdir_name": str(pointdir_name),
            "seed_frame_id": (
                int(seed_frame_id) if seed_frame_id is not None else None
            ),
        }
        data["records"].append(record)
        p = _index_path(campaign_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, data)
        return p


def _resolved_reference_entries(reference_data_dir: Union[str, Path]):
    from .reference_data import ReferenceDataVersioning

    base = Path(reference_data_dir)
    versioning = ReferenceDataVersioning(base)
    current = versioning.current_version()
    if current is None:
        return ()
    return versioning.resolve(current, verification="metadata").entries


def seed_frame_ids_from_committed_pointdirs(
    reference_data_dir: Union[str, Path],
) -> Set[int]:
    """Read seed frame IDs from the authoritative cumulative reference view."""
    out: Set[int] = set()
    for entry in _resolved_reference_entries(reference_data_dir):
        data = read_provenance(entry.pointdir_path)
        fid = (data.get("seed") or {}).get("frame_id")
        if isinstance(fid, int):
            out.add(int(fid))
    return out


def _records_from_committed_pointdirs(
    reference_data_dir: Union[str, Path],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in _resolved_reference_entries(reference_data_dir):
        data = read_provenance(entry.pointdir_path)
        fid = (data.get("seed") or {}).get("frame_id")
        out.append({
            "iteration": int(entry.introduced_in_version),
            "pointdir_name": entry.pointdir_name,
            "seed_frame_id": int(fid) if isinstance(fid, int) else None,
        })
    return out


def repair_index_from_committed_pointdirs(
    campaign_dir: Union[str, Path],
    reference_data_dir: Union[str, Path],
) -> int:
    """Append missing committed pointdir records to seed_frame_id_index.json.

    Existing records are keyed by (iteration, pointdir_name), making this
    helper idempotent. Returns the number of records added.
    """
    truth = _records_from_committed_pointdirs(reference_data_dir)
    if not truth:
        return 0
    with _index_lock(campaign_dir):
        data = load_index(campaign_dir)
        records = data.get("records", [])
        existing = {
            (int(rec.get("iteration")), str(rec.get("pointdir_name")))
            for rec in records
            if isinstance(rec, dict)
            and rec.get("iteration") is not None
            and rec.get("pointdir_name") is not None
        }
        added = 0
        for rec in truth:
            key = (int(rec["iteration"]), str(rec["pointdir_name"]))
            if key in existing:
                continue
            records.append(dict(rec))
            existing.add(key)
            added += 1
        if added:
            data["records"] = records
            p = _index_path(campaign_dir)
            p.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(p, data)
        return added


def load_training_seed_frame_ids(
    campaign_dir: Union[str, Path],
    reference_data_dir: Optional[Union[str, Path]] = None,
) -> Set[int]:
    """Return the set of stable trajectory frame_ids that have already been
    used as seeds for committed QM reference-data points.

    SEED_SELECT calls this once per iteration to compute its
    `forbidden_frame_ids` set (intersected with the recent-seeds cooldown).
    Frames whose `seed_frame_id` is ``None`` (synthetic / no-pool) are
    silently ignored.

    The flat index is the fast path, but a crash between commit() and the
    index-append loop can leave it short of what is actually committed. When
    `training_dir` is given we also scan the committed pointdir sidecars and
    union them in, so a truncated index can never make SEED_SELECT re-pick a
    frame that is already in the QM reference data. Reading only -- the index is not
    rewritten here.
    """
    data = load_index(campaign_dir)
    result: Set[int] = set()
    for rec in data.get("records", []):
        fid = rec.get("seed_frame_id")
        if isinstance(fid, int):
            result.add(int(fid))
    if reference_data_dir is not None:
        result |= seed_frame_ids_from_committed_pointdirs(reference_data_dir)
    return result


# ---------------------------------------------------------------------------
# recent-seeds cooldown cache -- the second anti-overlap rail.
#
# this is a small rolling cache of seed frame_ids for the last K iterations.
# SEED_SELECT reads it alongside the training-pool index to compute which
# frames we shouldn't pick from this round. the point is to stop adversarial
# descent from starting from the very same frame two iterations running --
# even if the trained model hasn't seen it yet (so the training-pool index
# wouldn't flag it), repeating the same starting point burns compute on
# essentially the same trajectory. the cache file lives next to the index
# and writes go through atomic_write_json so a half-written cache after a
# crash is impossible.

RECENT_SEEDS_FILENAME = "recent_seeds.json"
RECENT_SEEDS_SCHEMA_VERSION = 1
DEFAULT_RECENT_SEEDS_COOLDOWN = 3


def _recent_seeds_path(campaign_dir: Union[str, Path]) -> Path:
    return (
        Path(campaign_dir) / ".DATA" / "ACTIVE_LEARNING" / RECENT_SEEDS_FILENAME
    )


def _empty_recent_seeds_payload() -> Dict[str, Any]:
    return {
        "schema_version": RECENT_SEEDS_SCHEMA_VERSION,
        "cooldown": DEFAULT_RECENT_SEEDS_COOLDOWN,
        "history": [],
    }


def load_recent_seeds_payload(campaign_dir: Union[str, Path]) -> Dict[str, Any]:
    """Read the recent-seeds payload (or an empty one on missing file)."""
    p = _recent_seeds_path(campaign_dir)
    if not p.is_file():
        return _empty_recent_seeds_payload()
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ProvenanceError(
            "recent_seeds.json failed to parse: " + str(exc)
        ) from exc
    if not isinstance(data, dict):
        raise ProvenanceError("recent_seeds.json must be a JSON object")
    schema = int(data.get("schema_version", -1))
    if schema != RECENT_SEEDS_SCHEMA_VERSION:
        raise ProvenanceError(
            "recent_seeds.json schema_version " + str(schema)
            + " != " + str(RECENT_SEEDS_SCHEMA_VERSION)
        )
    if not isinstance(data.get("history"), list):
        raise ProvenanceError("recent_seeds.json `history` must be a list")
    return data


def load_recent_seed_frame_ids(campaign_dir: Union[str, Path]) -> Set[int]:
    """Return the union of seed_frame_ids in the rolling cooldown cache."""
    data = load_recent_seeds_payload(campaign_dir)
    result: Set[int] = set()
    for entry in data.get("history", []):
        for fid in entry.get("frame_ids", []):
            if isinstance(fid, int):
                result.add(int(fid))
    return result


def append_recent_seeds(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
    frame_ids: Sequence[Optional[int]],
    cooldown: int = DEFAULT_RECENT_SEEDS_COOLDOWN,
) -> Path:
    """Append one (iteration, frame_ids) record and trim to the last
    ``cooldown`` iterations. ``frame_ids`` may contain ``None`` entries;
    those are dropped. Returns the path of the atomic-write target."""
    # same flock pattern as the index file above -- two writers racing on
    # the cooldown cache would corrupt the history list.
    with _recent_seeds_lock(campaign_dir):
        data = load_recent_seeds_payload(campaign_dir)
        clean: List[int] = [int(f) for f in frame_ids if isinstance(f, int)]
        # if someone passes a negative cooldown, clamp to zero. python
        # slicing has a sharp edge here -- lst[-0:] returns the WHOLE
        # list, not the empty list you might guess. so without this clamp
        # a negative number silently meant "keep everything forever".
        cooldown = max(0, int(cooldown))
        data["cooldown"] = cooldown
        data["history"].append({
            "iteration": int(iteration),
            "frame_ids": clean,
        })
        if cooldown == 0:
            data["history"] = []
        elif len(data["history"]) > cooldown:
            data["history"] = data["history"][-cooldown:]
        p = _recent_seeds_path(campaign_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, data)
        return p
