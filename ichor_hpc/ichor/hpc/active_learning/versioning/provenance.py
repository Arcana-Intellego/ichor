"""Per-pointdir provenance ledger.

Every committed `.pointdir` in `QM_REFERENCE_DATA/iteration-NNNNNN/` carries a
`provenance.json` sidecar that traces the point back through the
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

from ..strict_json import strict_json as json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from contextlib import contextmanager

from ..daemon.state import atomic_write_json


# the read-modify-write helpers below need an actual lock, not just the
# atomic-rename trick atomic_write_json gives us. two enrich callers
# racing on the same provenance.json would both read the old file, each
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
    provenance.json. Lockfile lives inside the pointdir."""
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
    from ..daemon.filesystem import operational_data_dir

    campaign_dir = Path(campaign_dir)
    lock_dir = operational_data_dir(campaign_dir)
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
    from ..daemon.filesystem import operational_data_dir

    campaign_dir = Path(campaign_dir)
    lock_dir = operational_data_dir(campaign_dir)
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
    "upsert_index_records",
    "load_index",
    "load_training_seed_frame_ids",
    "repair_index_from_committed_pointdirs",
    "seed_frame_ids_from_committed_pointdirs",
    "append_recent_seeds",
    "load_recent_seeds_payload",
    "load_recent_seed_frame_ids",
]


PROVENANCE_FILENAME = "provenance.json"
PROVENANCE_SCHEMA_VERSION = 4

# Index lives under <campaign>/.DATA/ACTIVE_LEARNING/
SEED_FRAME_ID_INDEX_FILENAME = "seed_frame_id_index.json"
INDEX_SCHEMA_VERSION = 2


class ProvenanceError(RuntimeError):
    """Raised on malformed provenance JSON or index files."""


def _exact_integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProvenanceError(label + " must be an exact JSON integer")
    parsed = int(value)
    if parsed < int(minimum):
        raise ProvenanceError(label + " must be >= " + str(int(minimum)))
    return parsed


def _sha256_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProvenanceError(label + " must be a lowercase SHA-256 digest")
    return value


def _finite_number_or_none(value: Any, label: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(label + " must be a finite JSON number or null")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProvenanceError(label + " must be finite")
    return parsed


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
    seed_id: Optional[int] = None,
    seed_uid: Optional[str] = None,
    array_task_id_zero_based: Optional[int] = None,
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
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise ProvenanceError("provenance campaign_uid must be a non-empty string")
    iteration_value = _exact_integer(iteration, "provenance iteration")
    trajectory_digest = _sha256_text(
        trajectory_sha256, "provenance trajectory_sha256"
    )
    frame_id = (
        None
        if seed_frame_id is None
        else _exact_integer(seed_frame_id, "provenance seed.frame_id")
    )
    seed_id_value = (
        None
        if seed_id is None
        else _exact_integer(seed_id, "provenance seed.seed_id", minimum=1)
    )
    array_id_value = (
        None
        if array_task_id_zero_based is None
        else _exact_integer(
            array_task_id_zero_based,
            "provenance seed.array_task_id_zero_based",
        )
    )
    if iteration_value >= 1:
        if seed_id_value is None or array_id_value != seed_id_value - 1:
            raise ProvenanceError("active provenance seed/task identity is incomplete")
        _sha256_text(seed_uid, "provenance seed.seed_uid")
    elif seed_uid is not None:
        _sha256_text(seed_uid, "provenance seed.seed_uid")
    if not isinstance(seed_selection_origin, str) or not seed_selection_origin:
        raise ProvenanceError("provenance seed.selection_origin must be non-empty")
    variance_value = _finite_number_or_none(
        seed_variance_at_selection,
        "provenance seed.variance_at_selection",
    )
    dimension_value = _exact_integer(
        subspace_dimension, "provenance subspace.dimension"
    )
    neighbour_values = [
        _exact_integer(value, "provenance subspace.neighbour_frame_ids")
        for value in subspace_neighbour_frame_ids
    ]
    if len(neighbour_values) != len(set(neighbour_values)):
        raise ProvenanceError("provenance subspace neighbour IDs contain duplicates")
    eigenvalues = [
        _finite_number_or_none(value, "provenance subspace.eigenvalues")
        for value in subspace_eigenvalues
    ]
    if any(value is None or value < 0.0 for value in eigenvalues):
        raise ProvenanceError("provenance subspace eigenvalues must be non-negative")
    if len(eigenvalues) < dimension_value:
        raise ProvenanceError("provenance subspace has fewer eigenvalues than dimensions")
    if not isinstance(mode_weighting_policy, str) or not mode_weighting_policy:
        raise ProvenanceError("provenance mode_weighting_policy must be non-empty")

    payload: Dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "campaign_uid": campaign_uid,
        "iteration": iteration_value,
        "trajectory_sha256": trajectory_digest,
        "seed": {
            "seed_id": seed_id_value,
            "seed_uid": None if seed_uid is None else str(seed_uid),
            "array_task_id_zero_based": array_id_value,
            "frame_id": frame_id,
            "selection_origin": seed_selection_origin,
            "variance_at_selection": variance_value,
        },
        "subspace": {
            "neighbour_frame_ids": neighbour_values,
            "dimension": dimension_value,
            "eigenvalues": eigenvalues,
            "mode_weighting_policy": mode_weighting_policy,
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
    schema = _exact_integer(data.get("schema_version"), "provenance schema_version")
    if schema != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError(
            str(p) + ": schema_version " + str(schema)
            + " != " + str(PROVENANCE_SCHEMA_VERSION)
        )
    return data


def _expect_int_or_none(value: Any, label: str) -> Optional[int]:
    if value is None:
        return None
    return _exact_integer(value, label)


def validate_provenance(
    pointdir: Union[str, Path],
    *,
    campaign_uid: Optional[str] = None,
    iteration: Optional[int] = None,
    trajectory_sha256: Optional[str] = None,
    seed_frame_id: Optional[int] = None,
    seed_id: Optional[int] = None,
    seed_uid: Optional[str] = None,
    array_task_id_zero_based: Optional[int] = None,
    require_phase_b_selected: Optional[bool] = None,
    allocation_split: Optional[str] = None,
    allocation_slot_id: Optional[int] = None,
    allocation_candidate_id: Optional[str] = None,
    allocation_context: Optional[str] = None,
    allocation_slot_assignment_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Read provenance and validate the campaign/seed handoff contract."""
    data = read_provenance(pointdir)
    if not isinstance(data.get("campaign_uid"), str) or not data.get("campaign_uid"):
        raise ProvenanceError("provenance campaign_uid is missing or invalid")
    if campaign_uid is not None and str(data.get("campaign_uid")) != str(campaign_uid):
        raise ProvenanceError("provenance campaign_uid mismatch")
    prov_iteration = _exact_integer(data.get("iteration"), "provenance iteration")
    if iteration is not None and prov_iteration != int(iteration):
        raise ProvenanceError("provenance iteration mismatch")
    observed_trajectory_sha256 = _sha256_text(
        data.get("trajectory_sha256"), "provenance trajectory_sha256"
    )
    if (
        trajectory_sha256 is not None
        and observed_trajectory_sha256 != str(trajectory_sha256)
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
    _finite_number_or_none(
        seed.get("variance_at_selection"),
        "provenance seed.variance_at_selection",
    )
    observed_seed_id = _expect_int_or_none(
        seed.get("seed_id"),
        "provenance seed.seed_id",
    )
    observed_array_task_id = _expect_int_or_none(
        seed.get("array_task_id_zero_based"),
        "provenance seed.array_task_id_zero_based",
    )
    observed_seed_uid = seed.get("seed_uid")
    if prov_iteration >= 1:
        if observed_seed_id is None or observed_seed_id < 1:
            raise ProvenanceError("active provenance seed.seed_id must be >= 1")
        if observed_array_task_id != observed_seed_id - 1:
            raise ProvenanceError("active provenance array task/seed mismatch")
        _sha256_text(observed_seed_uid, "active provenance seed.seed_uid")
    if seed_id is not None and observed_seed_id != int(seed_id):
        raise ProvenanceError("provenance seed.seed_id mismatch")
    if seed_uid is not None and str(observed_seed_uid or "") != str(seed_uid):
        raise ProvenanceError("provenance seed.seed_uid mismatch")
    if (
        array_task_id_zero_based is not None
        and observed_array_task_id != int(array_task_id_zero_based)
    ):
        raise ProvenanceError("provenance seed.array_task_id_zero_based mismatch")

    subspace = data.get("subspace")
    if not isinstance(subspace, dict):
        raise ProvenanceError("provenance subspace block must be an object")
    dimension = _exact_integer(
        subspace.get("dimension"), "provenance subspace.dimension"
    )
    neighbours = subspace.get("neighbour_frame_ids")
    if not isinstance(neighbours, list):
        raise ProvenanceError("provenance subspace.neighbour_frame_ids must be a list")
    neighbour_values = [
        _exact_integer(value, "provenance subspace.neighbour_frame_ids")
        for value in neighbours
    ]
    if len(neighbour_values) != len(set(neighbour_values)):
        raise ProvenanceError("provenance subspace neighbour IDs contain duplicates")
    eigenvalues = subspace.get("eigenvalues")
    if not isinstance(eigenvalues, list):
        raise ProvenanceError("provenance subspace.eigenvalues must be a list")
    parsed_eigenvalues = [
        _finite_number_or_none(value, "provenance subspace.eigenvalues")
        for value in eigenvalues
    ]
    if any(value is None or value < 0.0 for value in parsed_eigenvalues):
        raise ProvenanceError("provenance subspace eigenvalues must be non-negative")
    if len(parsed_eigenvalues) < dimension:
        raise ProvenanceError("provenance subspace has fewer eigenvalues than dimensions")
    if not isinstance(subspace.get("mode_weighting_policy"), str) or not subspace.get(
        "mode_weighting_policy"
    ):
        raise ProvenanceError("provenance subspace mode-weighting policy is invalid")
    for optional_section in (
        "ariadne",
        "anti_overlap",
        "error_calibration_input",
        "phase_b",
        "point_allocation",
    ):
        section = data.get(optional_section)
        if section is not None and not isinstance(section, dict):
            raise ProvenanceError(
                "provenance " + optional_section + " must be an object or null"
            )

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
        or allocation_slot_assignment_sha256 is not None
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
        assignment_sha = str(allocation.get("slot_assignment_sha256") or "")
        if len(assignment_sha) != 64:
            raise ProvenanceError(
                "provenance point_allocation.slot_assignment_sha256 is invalid"
            )
        if (
            allocation_slot_assignment_sha256 is not None
            and assignment_sha != str(allocation_slot_assignment_sha256)
        ):
            raise ProvenanceError(
                "provenance point_allocation.slot_assignment_sha256 mismatch"
            )
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
    allocation_slot_assignment_sha256: Optional[str] = None,
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
    if allocation_slot_assignment_sha256 is not None:
        payload["slot_assignment_sha256"] = str(
            allocation_slot_assignment_sha256
        )
    return _merge_section(pointdir, "point_allocation", payload)


# ---------------------------------------------------------------------------
# Flat index file -- O(1) read at SEED_SELECT time

def _index_path(campaign_dir: Union[str, Path]) -> Path:
    from ..daemon.filesystem import operational_path

    return operational_path(campaign_dir, SEED_FRAME_ID_INDEX_FILENAME)


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
    schema = _exact_integer(
        data.get("schema_version"), "seed_frame_id_index.json schema_version"
    )
    if schema != INDEX_SCHEMA_VERSION:
        raise ProvenanceError(
            "seed_frame_id_index.json schema_version " + str(schema)
            + " != " + str(INDEX_SCHEMA_VERSION)
        )
    records = data.get("records")
    if not isinstance(records, list):
        raise ProvenanceError("seed_frame_id_index.json `records` must be a list")
    normalised = [_normalise_index_record(record) for record in records]
    keys = [(record["iteration"], record["pointdir_name"]) for record in normalised]
    if len(keys) != len(set(keys)):
        raise ProvenanceError("seed_frame_id_index.json contains duplicate point records")
    if keys != sorted(keys):
        raise ProvenanceError("seed_frame_id_index.json records are not canonically ordered")
    return {"schema_version": INDEX_SCHEMA_VERSION, "records": normalised}


def append_to_index(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
    pointdir_name: str,
    seed_frame_id: Optional[int],
    trajectory_sha256: str,
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
    return upsert_index_records(
        campaign_dir,
        records=[{
            "iteration": int(iteration),
            "pointdir_name": str(pointdir_name),
            "seed_frame_id": (
                int(seed_frame_id) if seed_frame_id is not None else None
            ),
            "trajectory_sha256": str(trajectory_sha256),
        }],
    )


def _normalise_index_record(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceError("seed-frame index records must be JSON objects")
    iteration = value.get("iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ProvenanceError("seed-frame index iteration must be a non-negative integer")
    pointdir_name = value.get("pointdir_name")
    if (
        not isinstance(pointdir_name, str)
        or not pointdir_name
        or Path(pointdir_name).name != pointdir_name
    ):
        raise ProvenanceError("seed-frame index pointdir_name must be one safe name")
    frame_id = value.get("seed_frame_id")
    if frame_id is not None and (
        isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0
    ):
        raise ProvenanceError(
            "seed-frame index seed_frame_id must be a non-negative integer or null"
        )
    trajectory_sha256 = _sha256_text(
        value.get("trajectory_sha256"),
        "seed-frame index trajectory_sha256",
    )
    return {
        "iteration": int(iteration),
        "pointdir_name": pointdir_name,
        "seed_frame_id": None if frame_id is None else int(frame_id),
        "trajectory_sha256": trajectory_sha256,
    }


def upsert_index_records(
    campaign_dir: Union[str, Path],
    *,
    records: Sequence[Dict[str, Any]],
) -> Path:
    """Atomically upsert one committed batch without replay duplication."""
    incoming = [_normalise_index_record(record) for record in records]
    with _index_lock(campaign_dir):
        data = load_index(campaign_dir)
        by_key: Dict[Tuple[int, str], Dict[str, Any]] = {}
        for raw in list(data.get("records") or []) + incoming:
            record = _normalise_index_record(raw)
            key = (record["iteration"], record["pointdir_name"])
            existing = by_key.get(key)
            if existing is not None and existing != record:
                raise ProvenanceError(
                    "conflicting seed-frame index record for "
                    + str(key[0])
                    + "/"
                    + key[1]
                )
            by_key[key] = record
        compact = [by_key[key] for key in sorted(by_key)]
        p = _index_path(campaign_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        if compact != data.get("records"):
            atomic_write_json(
                p,
                {
                    "schema_version": INDEX_SCHEMA_VERSION,
                    "records": compact,
                },
            )
        elif not p.is_file():
            atomic_write_json(p, _empty_index_payload())
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
    *,
    expected_trajectory_sha256: Optional[str] = None,
) -> Set[int]:
    """Read seed frame IDs from the authoritative cumulative reference view."""
    out: Set[int] = set()
    for entry in _resolved_reference_entries(reference_data_dir):
        data = read_provenance(entry.pointdir_path)
        observed_sha256 = _sha256_text(
            data.get("trajectory_sha256"),
            "committed provenance trajectory_sha256",
        )
        if (
            expected_trajectory_sha256 is not None
            and observed_sha256 != str(expected_trajectory_sha256)
        ):
            continue
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
            "trajectory_sha256": _sha256_text(
                data.get("trajectory_sha256"),
                "committed provenance trajectory_sha256",
            ),
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
    *,
    expected_trajectory_sha256: Optional[str] = None,
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
        if (
            expected_trajectory_sha256 is not None
            and rec["trajectory_sha256"] != str(expected_trajectory_sha256)
        ):
            continue
        fid = rec.get("seed_frame_id")
        if isinstance(fid, int):
            result.add(int(fid))
    if reference_data_dir is not None:
        result |= seed_frame_ids_from_committed_pointdirs(
            reference_data_dir,
            expected_trajectory_sha256=expected_trajectory_sha256,
        )
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
RECENT_SEEDS_SCHEMA_VERSION = 2
DEFAULT_RECENT_SEEDS_COOLDOWN = 3


def _recent_seeds_path(campaign_dir: Union[str, Path]) -> Path:
    from ..daemon.filesystem import operational_path

    return operational_path(campaign_dir, RECENT_SEEDS_FILENAME)


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
    schema = _exact_integer(
        data.get("schema_version"), "recent_seeds.json schema_version"
    )
    if schema != RECENT_SEEDS_SCHEMA_VERSION:
        raise ProvenanceError(
            "recent_seeds.json schema_version " + str(schema)
            + " != " + str(RECENT_SEEDS_SCHEMA_VERSION)
        )
    cooldown = _exact_integer(data.get("cooldown"), "recent_seeds.json cooldown")
    if not isinstance(data.get("history"), list):
        raise ProvenanceError("recent_seeds.json `history` must be a list")
    history = []
    previous_iteration = -1
    for raw_entry in data["history"]:
        if not isinstance(raw_entry, dict):
            raise ProvenanceError("recent_seeds.json history entries must be objects")
        iteration = _exact_integer(
            raw_entry.get("iteration"), "recent_seeds.json iteration"
        )
        if iteration <= previous_iteration:
            raise ProvenanceError(
                "recent_seeds.json history must have strictly increasing iterations"
            )
        previous_iteration = iteration
        frame_ids = raw_entry.get("frame_ids")
        if not isinstance(frame_ids, list):
            raise ProvenanceError("recent_seeds.json frame_ids must be a list")
        parsed_frame_ids = [
            _exact_integer(value, "recent_seeds.json frame_id")
            for value in frame_ids
        ]
        if len(parsed_frame_ids) != len(set(parsed_frame_ids)):
            raise ProvenanceError("recent_seeds.json contains duplicate frame IDs")
        history.append({
            "iteration": iteration,
            "frame_ids": parsed_frame_ids,
            "trajectory_sha256": _sha256_text(
                raw_entry.get("trajectory_sha256"),
                "recent_seeds.json trajectory_sha256",
            ),
        })
    if len(history) > cooldown:
        raise ProvenanceError("recent_seeds.json history exceeds its cooldown")
    return {
        "schema_version": RECENT_SEEDS_SCHEMA_VERSION,
        "cooldown": cooldown,
        "history": history,
    }


def load_recent_seed_frame_ids(
    campaign_dir: Union[str, Path],
    *,
    expected_trajectory_sha256: Optional[str] = None,
) -> Set[int]:
    """Return the union of seed_frame_ids in the rolling cooldown cache."""
    data = load_recent_seeds_payload(campaign_dir)
    result: Set[int] = set()
    for entry in data.get("history", []):
        if (
            expected_trajectory_sha256 is not None
            and entry["trajectory_sha256"] != str(expected_trajectory_sha256)
        ):
            continue
        for fid in entry.get("frame_ids", []):
            if isinstance(fid, int):
                result.add(int(fid))
    return result


def append_recent_seeds(
    campaign_dir: Union[str, Path],
    *,
    iteration: int,
    frame_ids: Sequence[Optional[int]],
    trajectory_sha256: str,
    cooldown: int = DEFAULT_RECENT_SEEDS_COOLDOWN,
) -> Path:
    """Append one (iteration, frame_ids) record and trim to the last
    ``cooldown`` iterations. ``frame_ids`` may contain ``None`` entries;
    those are dropped. Returns the path of the atomic-write target."""
    # same flock pattern as the index file above -- two writers racing on
    # the cooldown cache would corrupt the history list.
    with _recent_seeds_lock(campaign_dir):
        data = load_recent_seeds_payload(campaign_dir)
        clean: List[int] = []
        for frame_id in frame_ids:
            if frame_id is None:
                continue
            clean.append(_exact_integer(frame_id, "recent seed frame_id"))
        clean = sorted(set(clean))
        iteration_value = _exact_integer(iteration, "recent seed iteration")
        cooldown = _exact_integer(cooldown, "recent seed cooldown")
        trajectory_digest = _sha256_text(
            trajectory_sha256, "recent seed trajectory_sha256"
        )
        data["cooldown"] = cooldown
        history_by_iteration = {
            int(entry["iteration"]): dict(entry)
            for entry in data["history"]
        }
        new_entry = {
            "iteration": iteration_value,
            "frame_ids": clean,
            "trajectory_sha256": trajectory_digest,
        }
        existing = history_by_iteration.get(iteration_value)
        if existing is not None and existing != new_entry:
            raise ProvenanceError(
                "conflicting recent-seed record for iteration "
                + str(iteration_value)
            )
        history_by_iteration[iteration_value] = new_entry
        data["history"] = [
            history_by_iteration[key] for key in sorted(history_by_iteration)
        ]
        if cooldown == 0:
            data["history"] = []
        elif len(data["history"]) > cooldown:
            data["history"] = data["history"][-cooldown:]
        p = _recent_seeds_path(campaign_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, data)
        return p
