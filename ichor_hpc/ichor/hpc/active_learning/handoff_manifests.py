"""Strict JSON contracts between ARIADNE, Phase B diversity, and Gaussian.

These manifests are deliberately small and daemon-owned.  They record the
lineage that cannot be recovered safely from glob order once ARIADNE and
Phase B have filtered or reordered candidates.
"""
from __future__ import annotations

from .strict_json import strict_json as json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .daemon.state import atomic_write_json
from .layout import (
    active_ariadne_dir,
    active_iteration_dir,
    active_phase_b_dir,
    active_seed_selection_dir,
    ariadne_seeds_dir,
    bootstrap_selection_dir,
    parse_seed_directory_name,
)
from .selection_origins import (
    SEED_SELECTION_ORIGINS,
    SEED_SELECTION_READER_ORIGINS,
)


ARIADNE_RESULTS_FILENAME = "RESULTS.json"
ARIADNE_RESULTS_SCHEMA_VERSION = 3
ARIADNE_BATCH_DECISION_FILENAME = "ARIADNE_BATCH_DECISION.json"
ARIADNE_BATCH_DECISION_SCHEMA_VERSION = 2
ARIADNE_LANDING_AUDIT_FILENAME = "AUDIT.json"
ARIADNE_LANDING_AUDIT_SCHEMA_VERSION = 3
PHASE_A_SAMPLE_FILENAME = "SELECTION.json"
PHASE_A_SAMPLE_SCHEMA_VERSION = 5
PHASE_B_SELECTION_FILENAME = "SELECTION.json"
PHASE_B_SELECTION_SCHEMA_VERSION = 4
SEED_SELECTION_FILENAME = "SELECTION.json"
SEED_SELECTION_SCHEMA_VERSION = 3
SEED_SELECTION_DIAGNOSTICS_FILENAME = SEED_SELECTION_FILENAME
SEED_SELECTION_DIAGNOSTICS_SCHEMA_VERSION = SEED_SELECTION_SCHEMA_VERSION
ACQUISITION_MATURITY_AUDIT_FILENAME = ARIADNE_LANDING_AUDIT_FILENAME
ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION = ARIADNE_LANDING_AUDIT_SCHEMA_VERSION
ARIADNE_TASK_MAP_FILENAME = "TASK_MAP.json"
ARIADNE_TASK_MAP_SCHEMA_VERSION = 2


class HandoffManifestError(ValueError):
    """Raised when a phase handoff manifest is missing or violates contract."""


def _phase_a_campaign_root(selection_root: Any) -> Path:
    resolved = Path(selection_root).resolve()
    try:
        campaign = resolved.parents[2]
    except IndexError as exc:
        raise HandoffManifestError(
            "Phase A selection directory is outside the canonical campaign layout"
        ) from exc
    if bootstrap_selection_dir(campaign).resolve() != resolved:
        raise HandoffManifestError(
            "Phase A selection directory is outside .DATA/BOOTSTRAP/selection: "
            + str(resolved)
        )
    return campaign


def _ariadne_results_counts(payload: Dict[str, Any]) -> Tuple[int, int, int]:
    accepted = payload.get("accepted")
    rejected = payload.get("rejected")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise HandoffManifestError(
            "ARIADNE results accepted/rejected records must be lists"
        )
    expected = _required_int(payload.get("expected_n"), "ARIADNE expected_n")
    n_accepted = _required_int(
        payload.get("n_accepted"),
        "ARIADNE n_accepted",
    )
    n_rejected = _required_int(
        payload.get("n_rejected"),
        "ARIADNE n_rejected",
    )
    if (
        expected <= 0
        or n_accepted != len(accepted)
        or n_rejected != len(rejected)
        or n_accepted + n_rejected != expected
    ):
        raise HandoffManifestError("ARIADNE results counts are inconsistent")
    return expected, n_accepted, n_rejected


def iteration_dir(campaign_dir: Any, iteration: int) -> Path:
    return active_iteration_dir(campaign_dir, iteration)


def seeds_picked_path(iter_dir: Any) -> Path:
    return active_seed_selection_dir(iter_dir) / SEED_SELECTION_FILENAME


def seed_selection_diagnostics_path(iter_dir: Any) -> Path:
    return seeds_picked_path(iter_dir)


def acquisition_maturity_audit_path(iter_dir: Any) -> Path:
    return ariadne_landing_audit_path(iter_dir)


def ariadne_results_path(iter_dir: Any) -> Path:
    return active_ariadne_dir(iter_dir) / ARIADNE_RESULTS_FILENAME


def ariadne_batch_decision_path(iter_dir: Any) -> Path:
    return active_ariadne_dir(iter_dir) / ARIADNE_BATCH_DECISION_FILENAME


def write_ariadne_batch_decision(
    iter_dir: Any,
    *,
    campaign_uid: str,
    iteration: int,
    config_sha256: str,
    failure_threshold_fraction: float,
    expected_n: int,
    n_accepted: int,
    n_rejected: int,
    accepted: bool,
    reasons: Sequence[str],
) -> Path:
    from .daemon.completion_receipts import canonical_sha256
    from .versioning.manifest import sha256_file

    results_path = ariadne_results_path(iter_dir)
    if results_path.is_symlink() or not results_path.is_file():
        raise HandoffManifestError(
            "ARIADNE batch decision requires a regular RESULTS.json"
        )
    try:
        results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE RESULTS.json is unreadable") from exc
    if not isinstance(results_payload, dict) or str(
        results_payload.get("campaign_uid") or ""
    ) != str(campaign_uid):
        raise HandoffManifestError("ARIADNE batch decision/results campaign UID mismatch")
    if _required_int(
        results_payload.get("schema_version"),
        "ARIADNE results schema",
    ) != ARIADNE_RESULTS_SCHEMA_VERSION:
        raise HandoffManifestError("ARIADNE batch decision requires current RESULTS schema")
    if _required_int(
        results_payload.get("iteration"),
        "ARIADNE results iteration",
    ) != int(iteration):
        raise HandoffManifestError("ARIADNE batch decision/results iteration mismatch")
    result_expected, result_accepted, result_rejected = _ariadne_results_counts(
        results_payload
    )
    expected = int(expected_n)
    accepted_count = int(n_accepted)
    rejected_count = int(n_rejected)
    if expected <= 0 or accepted_count < 0 or rejected_count < 0:
        raise HandoffManifestError("ARIADNE batch decision counts are invalid")
    if accepted_count + rejected_count != expected:
        raise HandoffManifestError("ARIADNE batch decision counts do not cover expected tasks")
    if (expected, accepted_count, rejected_count) != (
        result_expected,
        result_accepted,
        result_rejected,
    ):
        raise HandoffManifestError(
            "ARIADNE batch decision counts do not match RESULTS.json"
        )
    threshold = float(failure_threshold_fraction)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise HandoffManifestError("ARIADNE failure threshold must be in [0, 1]")
    fraction = rejected_count / float(expected)
    policy_accepts = accepted_count > 0 and fraction <= threshold
    if bool(accepted) != bool(policy_accepts):
        raise HandoffManifestError(
            "ARIADNE batch decision accepted flag disagrees with its policy"
        )
    evaluation = {
        "config_sha256": str(config_sha256),
        "failure_threshold_fraction": threshold,
        "expected_n": expected,
        "n_accepted": accepted_count,
        "n_rejected": rejected_count,
        "rejection_fraction": fraction,
        "accepted": bool(accepted),
        "reasons": [str(reason) for reason in reasons],
    }
    evaluation["evaluation_sha256"] = canonical_sha256(evaluation)
    path = ariadne_batch_decision_path(iter_dir)
    evaluations: List[Dict[str, Any]] = []
    if path.is_file() and not path.is_symlink():
        existing = read_ariadne_batch_decision(
            iter_dir,
            expected_iteration=int(iteration),
            require_accepted=False,
            verify_current_config=False,
        )
        if str(existing["campaign_uid"]) != str(campaign_uid):
            raise HandoffManifestError("ARIADNE batch decision campaign UID changed")
        if str(existing["results"]["sha256"]) != sha256_file(results_path):
            raise HandoffManifestError("ARIADNE results changed since prior batch decision")
        evaluations = [dict(item) for item in existing["evaluations"]]
    if not any(
        str(item.get("evaluation_sha256")) == str(evaluation["evaluation_sha256"])
        for item in evaluations
    ):
        evaluation["evaluated_at_iso"] = datetime.now(timezone.utc).isoformat()
        evaluations.append(evaluation)
    payload = {
        "schema_version": ARIADNE_BATCH_DECISION_SCHEMA_VERSION,
        "campaign_uid": str(campaign_uid),
        "iteration": int(iteration),
        "results": {
            "path": ARIADNE_RESULTS_FILENAME,
            "size": int(results_path.stat().st_size),
            "sha256": sha256_file(results_path),
        },
        "evaluations": evaluations,
        "current_evaluation_sha256": str(evaluation["evaluation_sha256"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def read_ariadne_batch_decision(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    expected_campaign_uid: Optional[str] = None,
    expected_config_sha256: Optional[str] = None,
    expected_failure_threshold_fraction: Optional[float] = None,
    require_accepted: bool = True,
    verify_current_config: bool = True,
) -> Dict[str, Any]:
    from .daemon.completion_receipts import canonical_sha256
    from .versioning.manifest import sha256_file

    path = ariadne_batch_decision_path(iter_dir)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("ARIADNE batch decision missing: " + str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE batch decision unreadable: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise HandoffManifestError("ARIADNE batch decision must be an object")
    if _required_int(payload.get("schema_version"), "ARIADNE batch decision schema") != ARIADNE_BATCH_DECISION_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported ARIADNE batch decision schema")
    iteration = _required_int(payload.get("iteration"), "ARIADNE batch decision iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("ARIADNE batch decision iteration mismatch")
    campaign_uid = str(payload.get("campaign_uid") or "")
    if not campaign_uid:
        raise HandoffManifestError("ARIADNE batch decision campaign UID is missing")
    if expected_campaign_uid is not None and campaign_uid != str(expected_campaign_uid):
        raise HandoffManifestError("ARIADNE batch decision campaign UID mismatch")
    binding = payload.get("results")
    if not isinstance(binding, dict):
        raise HandoffManifestError("ARIADNE batch decision results binding is missing")
    results_path = resolve_handoff_path(
        active_ariadne_dir(iter_dir),
        binding.get("path", ""),
        kind="ARIADNE batch decision results",
    )
    if results_path != ariadne_results_path(iter_dir).resolve():
        raise HandoffManifestError("ARIADNE batch decision binds a noncanonical results path")
    if int(binding.get("size", -1)) != int(results_path.stat().st_size):
        raise HandoffManifestError("ARIADNE batch decision results size mismatch")
    if str(binding.get("sha256") or "") != sha256_file(results_path):
        raise HandoffManifestError("ARIADNE batch decision results hash mismatch")
    try:
        results_payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE batch decision results are unreadable") from exc
    if not isinstance(results_payload, dict) or str(
        results_payload.get("campaign_uid") or ""
    ) != campaign_uid:
        raise HandoffManifestError("ARIADNE batch decision/results campaign UID mismatch")
    if _required_int(
        results_payload.get("schema_version"),
        "ARIADNE results schema",
    ) != ARIADNE_RESULTS_SCHEMA_VERSION:
        raise HandoffManifestError("ARIADNE batch decision requires current RESULTS schema")
    if _required_int(
        results_payload.get("iteration"),
        "ARIADNE results iteration",
    ) != iteration:
        raise HandoffManifestError("ARIADNE batch decision/results iteration mismatch")
    result_counts = _ariadne_results_counts(results_payload)
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, list) or not evaluations:
        raise HandoffManifestError("ARIADNE batch decision evaluations are empty")
    by_digest: Dict[str, Dict[str, Any]] = {}
    for raw in evaluations:
        if not isinstance(raw, dict):
            raise HandoffManifestError("ARIADNE batch evaluation must be an object")
        evaluation = dict(raw)
        declared = str(evaluation.pop("evaluation_sha256", ""))
        evaluation.pop("evaluated_at_iso", None)
        actual = canonical_sha256(evaluation)
        if declared != actual or declared in by_digest:
            raise HandoffManifestError("ARIADNE batch evaluation digest is invalid")
        full = dict(raw)
        expected = _required_int(full.get("expected_n"), "ARIADNE decision expected_n")
        n_accepted = _required_int(full.get("n_accepted"), "ARIADNE decision n_accepted")
        n_rejected = _required_int(full.get("n_rejected"), "ARIADNE decision n_rejected")
        if expected <= 0 or n_accepted + n_rejected != expected:
            raise HandoffManifestError("ARIADNE decision counts are inconsistent")
        if (expected, n_accepted, n_rejected) != result_counts:
            raise HandoffManifestError(
                "ARIADNE decision counts do not match RESULTS.json"
            )
        threshold = _finite_float(full.get("failure_threshold_fraction"))
        if threshold is None or not 0.0 <= threshold <= 1.0:
            raise HandoffManifestError("ARIADNE decision threshold is invalid")
        fraction = n_rejected / float(expected)
        declared_fraction = _finite_float(full.get("rejection_fraction"))
        if declared_fraction is None or not math.isclose(
            declared_fraction,
            fraction,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise HandoffManifestError("ARIADNE decision rejection fraction is invalid")
        if bool(full.get("accepted", False)) != bool(
            n_accepted > 0 and fraction <= threshold
        ):
            raise HandoffManifestError(
                "ARIADNE decision accepted flag disagrees with its policy"
            )
        by_digest[declared] = full
    current_digest = str(payload.get("current_evaluation_sha256") or "")
    if current_digest not in by_digest:
        raise HandoffManifestError("ARIADNE current batch evaluation is missing")
    current = by_digest[current_digest]
    if verify_current_config and expected_config_sha256 is not None and str(
        current.get("config_sha256") or ""
    ) != str(expected_config_sha256):
        raise HandoffManifestError("ARIADNE batch decision config digest mismatch")
    if (
        verify_current_config
        and expected_failure_threshold_fraction is not None
        and not math.isclose(
            float(current.get("failure_threshold_fraction")),
            float(expected_failure_threshold_fraction),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise HandoffManifestError(
            "ARIADNE batch decision failure threshold mismatch"
        )
    if require_accepted and not bool(current.get("accepted", False)):
        raise HandoffManifestError(
            "ARIADNE batch decision rejected: "
            + ";".join(str(reason) for reason in current.get("reasons", []))
        )
    out = dict(payload)
    out["results"] = {**binding, "path": str(results_path)}
    out["current_evaluation"] = current
    return out


def read_authoritative_ariadne_batch_decision(
    campaign_dir: Any,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
    require_accepted: bool = True,
) -> Dict[str, Any]:
    """Read a batch decision against its frozen producer contract."""
    from .config import CampaignConfig
    from .daemon.ariadne_decision_authority import (
        resolve_ariadne_handoff_decision_contract,
    )
    from .daemon.config_lock import canonical_config, config_fingerprint
    from .layout import active_iteration_dir
    from .seed_identity import read_ariadne_task_map

    campaign = Path(campaign_dir).expanduser().resolve()
    iter_dir = active_iteration_dir(campaign, int(iteration))
    task_map = read_ariadne_task_map(
        iter_dir,
        expected_iteration=int(iteration),
    )
    campaign_uid = str(
        expected_campaign_uid or task_map.get("campaign_uid") or ""
    )
    if not campaign_uid:
        raise HandoffManifestError(
            "ARIADNE task map campaign UID is missing"
        )
    if str(task_map.get("campaign_uid") or "") != campaign_uid:
        raise HandoffManifestError(
            "ARIADNE task map campaign UID mismatch"
        )
    try:
        authority = resolve_ariadne_handoff_decision_contract(
            campaign,
            campaign_uid=campaign_uid,
            iteration=int(iteration),
            logical_total=int(task_map["n_tasks"]),
            replacement_round=0,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HandoffManifestError(
            "ARIADNE producer decision contract is invalid: "
            + type(exc).__name__
            + ": "
            + str(exc)
        ) from exc
    if authority is None:
        config = CampaignConfig.from_yaml(campaign / "campaign.yaml")
        contract = {
            "config_sha256": config_fingerprint(canonical_config(config)),
            "failure_threshold_fraction": float(
                config.runtime.failure_threshold_fraction
            ),
        }
    else:
        contract = dict(authority["decision_contract"])
    return read_ariadne_batch_decision(
        iter_dir,
        expected_iteration=int(iteration),
        expected_campaign_uid=campaign_uid,
        expected_config_sha256=str(contract["config_sha256"]),
        expected_failure_threshold_fraction=float(
            contract["failure_threshold_fraction"]
        ),
        require_accepted=bool(require_accepted),
        verify_current_config=True,
    )


def ariadne_landing_audit_path(iter_dir: Any) -> Path:
    return active_ariadne_dir(iter_dir) / ARIADNE_LANDING_AUDIT_FILENAME


def ariadne_task_map_path(iter_dir: Any) -> Path:
    return active_ariadne_dir(iter_dir) / ARIADNE_TASK_MAP_FILENAME


def phase_a_sample_manifest_path(initial_dir: Any) -> Path:
    return Path(initial_dir) / PHASE_A_SAMPLE_FILENAME


def phase_b_selection_path(iter_dir: Any) -> Path:
    return active_phase_b_dir(iter_dir) / PHASE_B_SELECTION_FILENAME


def _finite_float(value: Any, *, allow_none: bool = False) -> Optional[float]:
    if value is None and allow_none:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise HandoffManifestError("expected finite numeric value, got " + repr(value)) from exc
    if not math.isfinite(out):
        raise HandoffManifestError("expected finite numeric value, got " + repr(value))
    return out


def _int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise HandoffManifestError(
            "expected exact JSON integer or null, got " + repr(value)
        )
    return int(value)


def _required_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HandoffManifestError(label + " must be an exact JSON integer")
    return int(value)


def _json_number_or_none(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _xyz_frame_count(path: Path, label: str) -> int:
    from ichor.core.files.xyz.strict_xyz import read_xyz_frames

    try:
        frames = read_xyz_frames(path)
    except (OSError, ValueError) as exc:
        raise HandoffManifestError(label + " is unreadable: " + str(path)) from exc
    if any(len(frame) <= 0 for frame in frames):
        raise HandoffManifestError(label + " contains an empty frame")
    return len(frames)


def resolve_handoff_path(
    root: Any,
    raw: Any,
    *,
    kind: str,
    must_exist: bool = True,
    directory: bool = False,
) -> Path:
    base = Path(root).resolve()
    p = Path(str(raw))
    if not p.is_absolute():
        p = base / p
    try:
        resolved = p.resolve(strict=False)
    except OSError as exc:
        raise HandoffManifestError(kind + " path cannot be resolved: " + str(p)) from exc
    if resolved != base and base not in resolved.parents:
        raise HandoffManifestError(kind + " path escapes handoff root: " + str(p))
    if p.is_symlink() or resolved.is_symlink():
        raise HandoffManifestError(kind + " path is a symlink: " + str(p))
    if must_exist:
        if directory:
            if not resolved.is_dir():
                raise FileNotFoundError(kind + " directory missing: " + str(resolved))
        elif not resolved.is_file():
            raise FileNotFoundError(kind + " file missing: " + str(resolved))
    return resolved


def _resolve_rejected_handoff_reference(
    root: Any,
    raw: Any,
    *,
    kind: str,
    expected: Path,
    directory: bool = False,
) -> Path:
    """Validate a rejected-task path without requiring rejected output."""
    resolved = resolve_handoff_path(
        root,
        raw,
        kind=kind,
        must_exist=False,
        directory=directory,
    )
    expected_resolved = Path(expected).resolve(strict=False)
    if resolved != expected_resolved:
        raise HandoffManifestError(kind + " does not match its canonical task path")
    if resolved.exists():
        if directory and not resolved.is_dir():
            raise HandoffManifestError(kind + " is not a directory: " + str(resolved))
        if not directory and not resolved.is_file():
            raise HandoffManifestError(kind + " is not a regular file: " + str(resolved))
    return resolved


def build_seed_selection_manifest(
    *,
    campaign_uid: str,
    campaign_random_seed: int,
    iteration: int,
    models_version: int,
    model_manifest_sha256: str,
    model_set_sha256: str,
    trajectory_sha256: str,
    selection_strategy: str,
    seed_records: Sequence[Dict[str, Any]],
    prior_mean_contract_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the canonical schema-v3 seed-selection handoff."""
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise HandoffManifestError("seed selection campaign_uid is empty")
    iteration_value = _required_int(iteration, "seed selection iteration")
    model_version = _required_int(models_version, "seed selection models_version")
    if iteration_value < 1 or model_version < 0:
        raise HandoffManifestError("seed selection iteration/model version is invalid")
    if selection_strategy not in {"hybrid_variance", "d_optimal"}:
        raise HandoffManifestError("seed selection strategy is invalid")
    for label, digest in (
        ("model_manifest_sha256", model_manifest_sha256),
        ("model_set_sha256", model_set_sha256),
        ("trajectory_sha256", trajectory_sha256),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise HandoffManifestError("seed selection " + label + " is invalid")
    if prior_mean_contract_sha256 is not None and (
        not isinstance(prior_mean_contract_sha256, str)
        or len(prior_mean_contract_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in prior_mean_contract_sha256
        )
    ):
        raise HandoffManifestError("seed selection prior-mean digest is invalid")

    records: List[Dict[str, Any]] = []
    for expected_seed_id, raw in enumerate(seed_records, start=1):
        if not isinstance(raw, dict):
            raise HandoffManifestError("seed selection records must be objects")
        record = dict(raw)
        seed_id = _required_int(record.get("seed_id"), "seed selection seed_id")
        if seed_id != expected_seed_id:
            raise HandoffManifestError("seed IDs must be contiguous from one")
        frame_id = _int_or_none(record.get("frame_id"))
        pool_row = _required_int(
            record.get("pool_row_index_zero_based"),
            "seed selection pool row",
        )
        if (frame_id is not None and frame_id < 0) or pool_row < 0:
            raise HandoffManifestError("seed selection frame/pool row is negative")
        origin = record.get("selection_origin")
        if origin not in SEED_SELECTION_ORIGINS:
            raise HandoffManifestError("seed selection origin is invalid")
        record["seed_id"] = seed_id
        record["frame_id"] = frame_id
        record["pool_row_index_zero_based"] = pool_row
        record["variance_at_selection"] = _finite_float(
            record.get("variance_at_selection"),
            allow_none=True,
        )
        record.pop("seed_uid", None)
        records.append(record)
    if not records:
        raise HandoffManifestError("seed selection cannot be empty")
    if len({record["pool_row_index_zero_based"] for record in records}) != len(records):
        raise HandoffManifestError("seed selection contains duplicate pool rows")

    from .randomness import derive_rng_seed

    randomness = derive_rng_seed(
        campaign_uid=campaign_uid,
        campaign_random_seed=campaign_random_seed,
        iteration=iteration_value,
        phase="SEED_SELECT",
        logical_task_id="seed-batch",
        random_purpose="bulk-seed-selection",
    )
    rows_by_origin = {
        origin: [
            record["pool_row_index_zero_based"]
            for record in records
            if record["selection_origin"] == origin
        ]
        for origin in ("bulk", "variance", "d_optimal", "d_optimal_backfill")
    }
    ranked = (
        rows_by_origin["variance"]
        if selection_strategy == "hybrid_variance"
        else rows_by_origin["d_optimal"] + rows_by_origin["d_optimal_backfill"]
    )
    payload: Dict[str, Any] = {
        "schema_version": SEED_SELECTION_SCHEMA_VERSION,
        "campaign_uid": campaign_uid,
        "iteration": iteration_value,
        "models_version": model_version,
        "model_manifest_sha256": model_manifest_sha256,
        "model_set_sha256": model_set_sha256,
        "trajectory_sha256": trajectory_sha256,
        "selection_strategy": selection_strategy,
        "n_picked": len(records),
        "frame_ids": [record["frame_id"] for record in records],
        "indices": [record["pool_row_index_zero_based"] for record in records],
        "bulk_indices": rows_by_origin["bulk"],
        "variance_indices": ranked,
        "d_optimal_indices": (
            rows_by_origin["d_optimal"] + rows_by_origin["d_optimal_backfill"]
        ),
        "d_optimal_backfill_indices": rows_by_origin["d_optimal_backfill"],
        "variances": [record["variance_at_selection"] for record in records],
        "seed_records": records,
        "randomness": randomness.to_dict(),
    }
    if prior_mean_contract_sha256 is not None:
        payload["prior_mean_contract_sha256"] = prior_mean_contract_sha256
        for record in records:
            record["prior_mean_contract_sha256"] = prior_mean_contract_sha256

    from .seed_identity import deterministic_seed_uid, selection_fingerprint_sha256

    fingerprint = selection_fingerprint_sha256(payload)
    payload["selection_fingerprint_sha256"] = fingerprint
    for record in records:
        record["seed_uid"] = deterministic_seed_uid(
            campaign_uid=campaign_uid,
            iteration=iteration_value,
            seed_id=record["seed_id"],
            frame_id=record["frame_id"],
            models_version=model_version,
            model_set_sha256=model_set_sha256,
            selection_fingerprint_sha256_value=fingerprint,
        )
    return payload


def load_seeds_picked(iter_dir: Any, *, expected_iteration: Optional[int] = None) -> Dict[str, Any]:
    """Read the authoritative one-based seed-selection manifest."""
    path = seeds_picked_path(iter_dir)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("seed selection manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("seed selection manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("seed selection manifest must be a JSON object")
    if _required_int(data.get("schema_version"), "seed selection schema_version") != SEED_SELECTION_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported seed selection schema")
    iteration = _required_int(data.get("iteration"), "seed selection iteration")
    if iteration < 1:
        raise HandoffManifestError("active seed selection iteration must be >= 1")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError(
            "seed selection iteration mismatch: expected "
            + str(int(expected_iteration))
            + " got "
            + str(iteration)
        )
    campaign_uid = data.get("campaign_uid")
    if not isinstance(campaign_uid, str) or not campaign_uid:
        raise HandoffManifestError("seed selection campaign_uid is empty")
    from .randomness import validate_rng_derivation

    try:
        randomness = validate_rng_derivation(
            data.get("randomness"),
            expected_campaign_uid=campaign_uid,
            expected_iteration=iteration,
            expected_phase="SEED_SELECT",
            expected_logical_task_id="seed-batch",
            expected_random_purpose="bulk-seed-selection",
        )
    except ValueError as exc:
        raise HandoffManifestError(
            "seed selection randomness contract is invalid: " + str(exc)
        ) from exc
    models_version = _required_int(
        data.get("models_version"),
        "seed selection models_version",
    )
    if models_version < 0:
        raise HandoffManifestError("seed selection models_version must be >= 0")
    strategy = data.get("selection_strategy")
    if strategy not in {"hybrid_variance", "d_optimal"}:
        raise HandoffManifestError("seed selection strategy is invalid")
    for key in ("model_manifest_sha256", "model_set_sha256", "trajectory_sha256"):
        value = str(data.get(key) or "")
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise HandoffManifestError("seed selection " + key + " is invalid")
    n_picked = _required_int(data.get("n_picked"), "seed selection n_picked")
    if n_picked < 1:
        raise HandoffManifestError("seed selection n_picked must be >= 1")
    raw_records = data.get("seed_records")
    if not isinstance(raw_records, list) or len(raw_records) != n_picked:
        raise HandoffManifestError("seed_records must cover every selected seed")
    records = []
    for expected_seed_id, raw in enumerate(raw_records, start=1):
        if not isinstance(raw, dict):
            raise HandoffManifestError("seed_records entries must be objects")
        seed_id = _required_int(raw.get("seed_id"), "seed_id")
        if seed_id != expected_seed_id:
            raise HandoffManifestError("seed IDs must be contiguous from one")
        pool_row = _required_int(
            raw.get("pool_row_index_zero_based"),
            "pool_row_index_zero_based",
        )
        if pool_row < 0:
            raise HandoffManifestError("pool_row_index_zero_based must be >= 0")
        frame_id = _int_or_none(raw.get("frame_id"))
        if frame_id is not None and frame_id < 0:
            raise HandoffManifestError("frame_id must be >= 0 or null")
        origin = str(raw.get("selection_origin") or "unknown")
        if origin not in SEED_SELECTION_READER_ORIGINS:
            raise HandoffManifestError("unknown selection_origin: " + origin)
        record = dict(raw)
        seed_uid = str(raw.get("seed_uid") or "")
        if len(seed_uid) != 64 or any(ch not in "0123456789abcdef" for ch in seed_uid):
            raise HandoffManifestError("seed_uid is invalid")
        record.update(
            {
                "seed_id": seed_id,
                "seed_uid": seed_uid,
                "frame_id": frame_id,
                "pool_row_index_zero_based": pool_row,
                "selection_origin": origin,
                "variance_at_selection": _finite_float(
                    raw.get("variance_at_selection"),
                    allow_none=True,
                ),
            }
        )
        records.append(record)
    record_pool_rows = [record["pool_row_index_zero_based"] for record in records]
    if len(set(record_pool_rows)) != len(record_pool_rows):
        raise HandoffManifestError("seed selection contains duplicate pool rows")
    record_frame_ids = [record["frame_id"] for record in records]
    raw_selected_indices = data.get("indices")
    if not isinstance(raw_selected_indices, list):
        raise HandoffManifestError("seed selection indices must be a list")
    selected_indices = [
        _required_int(value, "seed selection indices")
        for value in raw_selected_indices
    ]
    if selected_indices != record_pool_rows:
        raise HandoffManifestError("seed selection indices disagree with seed_records")
    raw_frame_ids = data.get("frame_ids")
    if not isinstance(raw_frame_ids, list):
        raise HandoffManifestError("seed selection frame_ids must be a list")
    selected_frame_ids = [
        _int_or_none(value)
        for value in raw_frame_ids
    ]
    if selected_frame_ids != record_frame_ids:
        raise HandoffManifestError("seed selection frame_ids disagree with seed_records")
    for key in (
        "bulk_indices",
        "variance_indices",
        "d_optimal_indices",
        "d_optimal_backfill_indices",
    ):
        raw_indices = data.get(key)
        if not isinstance(raw_indices, list):
            raise HandoffManifestError("seed selection " + key + " must be a list")
        parsed_indices = [_required_int(value, "seed selection " + key) for value in raw_indices]
        if any(value < 0 for value in parsed_indices):
            raise HandoffManifestError("seed selection " + key + " contains a negative row")
        if len(set(parsed_indices)) != len(parsed_indices):
            raise HandoffManifestError("seed selection " + key + " contains duplicates")
        if not set(parsed_indices).issubset(set(record_pool_rows)):
            raise HandoffManifestError("seed selection " + key + " is not a selected-row subset")
        data[key] = parsed_indices
    origin_rows = {
        origin: {
            record["pool_row_index_zero_based"]
            for record in records
            if record["selection_origin"] == origin
        }
        for origin in ("bulk", "variance", "d_optimal", "d_optimal_backfill")
    }
    if set(data["bulk_indices"]) != origin_rows["bulk"]:
        raise HandoffManifestError("seed selection bulk_indices disagree with origins")
    expected_ranked_rows = (
        origin_rows["variance"]
        if strategy == "hybrid_variance"
        else origin_rows["d_optimal"] | origin_rows["d_optimal_backfill"]
    )
    if set(data["variance_indices"]) != expected_ranked_rows:
        raise HandoffManifestError("seed selection variance_indices disagree with origins")
    if set(data["d_optimal_indices"]) != (
        origin_rows["d_optimal"] | origin_rows["d_optimal_backfill"]
    ):
        raise HandoffManifestError("seed selection d_optimal_indices disagree with origins")
    if set(data["d_optimal_backfill_indices"]) != origin_rows["d_optimal_backfill"]:
        raise HandoffManifestError(
            "seed selection d_optimal_backfill_indices disagree with origins"
        )
    from .seed_identity import deterministic_seed_uid, selection_fingerprint_sha256

    observed_fingerprint = data.get("selection_fingerprint_sha256")
    expected_fingerprint = selection_fingerprint_sha256(data)
    if observed_fingerprint != expected_fingerprint:
        raise HandoffManifestError("seed selection fingerprint mismatch")
    for record in records:
        expected_seed_uid = deterministic_seed_uid(
            campaign_uid=campaign_uid,
            iteration=iteration,
            seed_id=record["seed_id"],
            frame_id=record["frame_id"],
            models_version=models_version,
            model_set_sha256=str(data["model_set_sha256"]),
            selection_fingerprint_sha256_value=expected_fingerprint,
        )
        if record["seed_uid"] != expected_seed_uid:
            raise HandoffManifestError("seed selection seed_uid mismatch")
    out = dict(data)
    out["iteration"] = iteration
    out["models_version"] = models_version
    out["selection_strategy"] = strategy
    out["n_picked"] = n_picked
    out["indices"] = selected_indices
    out["frame_ids"] = selected_frame_ids
    out["randomness"] = randomness.to_dict()
    out["seed_records"] = records
    return out


def write_seed_selection_diagnostics(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    path = seeds_picked_path(iter_dir)
    data = load_seeds_picked(iter_dir)
    data["diagnostics"] = dict(payload)
    atomic_write_json(path, data)
    return path


def read_seed_selection_diagnostics(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    selection = load_seeds_picked(
        iter_dir,
        expected_iteration=expected_iteration,
    )
    diagnostics = selection.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise HandoffManifestError("seed selection diagnostics are missing")
    return dict(diagnostics)


def write_acquisition_maturity_audit(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    path = acquisition_maturity_audit_path(iter_dir)
    data: Dict[str, Any] = {
        "schema_version": ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION,
        "iteration": int(payload.get("iteration")),
        "summary": {},
        "seeds": [],
    }
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError(
                "ARIADNE audit is unreadable: " + str(path)
            ) from exc
        if not isinstance(existing, dict):
            raise HandoffManifestError("ARIADNE audit must be a JSON object")
        data.update(existing)
    data["schema_version"] = ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION
    data["iteration"] = int(payload.get("iteration"))
    data["maturity"] = {
        key: value for key, value in payload.items() if key != "iteration"
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return path


def acquisition_maturity_audit_payload(
    *,
    iteration: int,
    seed_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback_counts: Dict[str, int] = {}
    n_candidates = 0
    n_with_spectral = 0
    n_with_residual = 0
    n_with_banded_energy = 0
    seeds: List[Dict[str, Any]] = []
    for record in seed_records:
        candidates = []
        for candidate in list(record.get("landing_candidates") or []):
            if not isinstance(candidate, dict):
                continue
            metrics = candidate.get("metrics") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            n_candidates += 1
            if metrics.get("spectral_frequency_risk") is not None:
                n_with_spectral += 1
            if metrics.get("fullspace_residual_distance") is not None:
                n_with_residual += 1
            if metrics.get("banded_energy_risk") is not None:
                n_with_banded_energy += 1
            reasons = metrics.get("acquisition_fallback_reasons") or []
            if isinstance(reasons, list):
                for reason in reasons:
                    key = str(reason)
                    fallback_counts[key] = int(fallback_counts.get(key, 0)) + 1
            candidates.append({
                "candidate_index": candidate.get("candidate_index"),
                "origin": candidate.get("origin"),
                "accepted": bool(candidate.get("accepted", False)),
                "reasons": list(candidate.get("reasons") or []),
                "record_only_reasons": list(candidate.get("record_only_reasons") or []),
                "alpha": _json_number_or_none(candidate.get("alpha")),
                "informativeness_score": _json_number_or_none(candidate.get("informativeness_score")),
                "risk_penalty_score": _json_number_or_none(candidate.get("risk_penalty_score")),
                "metrics": {
                    key: (
                        list(metrics.get(key) or [])
                        if key == "acquisition_fallback_reasons"
                        else _json_number_or_none(metrics.get(key))
                    )
                    for key in (
                        "total_score",
                        "observable_score",
                        "outlier_penalty_score",
                        "energy_risk",
                        "banded_energy_risk",
                        "spectral_frequency_risk",
                        "legacy_frequency_risk",
                        "fullspace_residual_distance",
                        "fullspace_residual_penalty",
                        "aligned_rmsd_ang",
                        "aligned_rmsd_penalty",
                        "chemistry_penalty",
                        "distance_penalty",
                        "whitened_distance",
                        "acquisition_fallback_reasons",
                    )
                    if key in metrics
                },
            })
        selection_diagnostics = record.get("selection_diagnostics")
        if not isinstance(selection_diagnostics, dict):
            selection_diagnostics = None
        seeds.append({
            "seed_id": record.get("seed_id"),
            "seed_uid": record.get("seed_uid"),
            "result_json": record.get("result_json"),
            "landing_policy": (
                (record.get("landing_safety") or {}).get("policy")
                if isinstance(record.get("landing_safety"), dict)
                else None
            ),
            "landing_safety_metrics": (
                dict((record.get("landing_safety") or {}).get("metrics") or {})
                if isinstance(record.get("landing_safety"), dict)
                else {}
            ),
            "selection_diagnostics": selection_diagnostics,
            "landing_candidates": candidates,
        })
    return {
        "iteration": int(iteration),
        "summary": {
            "n_seeds": int(len(seed_records)),
            "n_candidates": int(n_candidates),
            "n_candidates_with_spectral": int(n_with_spectral),
            "n_candidates_with_fullspace_residual": int(n_with_residual),
            "n_candidates_with_banded_energy": int(n_with_banded_energy),
            "fallback_reasons": fallback_counts,
        },
        "seeds": seeds,
    }


def read_acquisition_maturity_audit(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = acquisition_maturity_audit_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("acquisition maturity audit missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("acquisition maturity audit unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("acquisition maturity audit must be a JSON object")
    if _required_int(data.get("schema_version", -1), "acquisition maturity audit schema_version") != ACQUISITION_MATURITY_AUDIT_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported acquisition maturity audit schema")
    iteration = _required_int(data.get("iteration"), "acquisition maturity audit iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("acquisition maturity audit iteration mismatch")
    maturity = data.get("maturity")
    if not isinstance(maturity, dict):
        raise HandoffManifestError("ARIADNE maturity audit section is missing")
    seeds = maturity.get("seeds")
    if not isinstance(seeds, list):
        raise HandoffManifestError("acquisition maturity audit seeds must be a list")
    return data


def validate_ariadne_result(
    result: Dict[str, Any],
    *,
    expected_iteration: int,
    seed_record: Dict[str, Any],
    expected_atom_types: Optional[Sequence[str]] = None,
    expected_initial_coordinates: Optional[Sequence[Sequence[float]]] = None,
    expected_trajectory_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise HandoffManifestError("ARIADNE result must be a JSON object")
    seed_id = _required_int(seed_record.get("seed_id"), "ARIADNE seed_record seed_id")
    if seed_id < 1:
        raise HandoffManifestError("ARIADNE seed_id must be >= 1")
    seed_uid = str(seed_record.get("seed_uid") or "")
    if len(seed_uid) != 64:
        raise HandoffManifestError("ARIADNE seed_uid is invalid")
    frame_id = seed_record.get("frame_id")
    if _required_int(result.get("iteration"), "ARIADNE result iteration") != int(expected_iteration):
        raise HandoffManifestError("wrong_iteration")
    if _required_int(result.get("seed_id"), "ARIADNE result seed_id") != seed_id:
        raise HandoffManifestError("wrong_seed_id")
    if str(result.get("seed_uid") or "") != seed_uid:
        raise HandoffManifestError("wrong_seed_uid")
    if _required_int(result.get("array_task_id"), "ARIADNE array_task_id") != seed_id - 1:
        raise HandoffManifestError("wrong_array_task_id")
    result_frame_id = _int_or_none(result.get("seed_frame_id"))
    if result_frame_id != frame_id:
        raise HandoffManifestError("wrong_seed_frame")
    result_sha = result.get("trajectory_sha256")
    if expected_trajectory_sha256:
        if str(result_sha or "") != str(expected_trajectory_sha256):
            raise HandoffManifestError("wrong_trajectory_sha256")
    return_code = _required_int(result.get("return_code"), "ARIADNE result return_code")
    try:
        from .acquisition.ariadne_runner import ariadne_result_usability_payload

        usability = ariadne_result_usability_payload(result)
    except Exception:
        usability = {"usable": False, "reason": "usability_check_failed"}
    if not bool(usability.get("usable", False)):
        raise HandoffManifestError(
            "ariadne_unusable:"
            + str(usability.get("reason", "unusable_landing"))
        )
    alpha_initial = _finite_float(result.get("alpha_initial"))
    alpha_final = _finite_float(result.get("alpha_final"))
    n_evaluations = _required_int(result.get("n_evaluations"), "ARIADNE result n_evaluations")
    wall_seconds = _finite_float(result.get("wall_seconds"))
    atom_types = result.get("atom_types")
    final_coordinates = result.get("final_coordinates")
    if not isinstance(atom_types, list) or not atom_types:
        raise HandoffManifestError("missing_atom_types")
    clean_atom_types = [str(x) for x in atom_types]
    if expected_atom_types is not None:
        expected = [str(x) for x in expected_atom_types]
        if clean_atom_types != expected:
            raise HandoffManifestError("atom_type_order_mismatch")
    if not isinstance(final_coordinates, list) or len(final_coordinates) != len(atom_types):
        raise HandoffManifestError("final_coordinates_shape_mismatch")
    clean_coords = []
    for row in final_coordinates:
        if not isinstance(row, list) or len(row) != 3:
            raise HandoffManifestError("final_coordinates_shape_mismatch")
        clean_coords.append([_finite_float(row[0]), _finite_float(row[1]), _finite_float(row[2])])
    if expected_initial_coordinates is not None:
        expected_initial = np.asarray(expected_initial_coordinates, dtype=float)
        if (
            expected_initial.shape != (len(clean_atom_types), 3)
            or not np.isfinite(expected_initial).all()
        ):
            raise HandoffManifestError("authoritative_seed_coordinates_invalid")
        for key in ("initial_coordinates", "seed_coordinates"):
            value = result.get(key)
            try:
                observed = np.asarray(value, dtype=float)
            except (TypeError, ValueError) as exc:
                raise HandoffManifestError(key + "_invalid") from exc
            if observed.shape != expected_initial.shape or not np.isfinite(observed).all():
                raise HandoffManifestError(key + "_shape_or_finiteness_mismatch")
            if not np.allclose(observed, expected_initial, atol=1.0e-10, rtol=0.0):
                raise HandoffManifestError(key + "_authoritative_seed_mismatch")
    alpha_trajectory = result.get("alpha_trajectory")
    if not isinstance(alpha_trajectory, list):
        raise HandoffManifestError("alpha_trajectory_missing")
    for value in alpha_trajectory:
        _finite_float(value)
    whitened = _finite_float(result.get("whitened_distance_final"), allow_none=True)
    return {
        "seed_id": seed_id,
        "seed_uid": seed_uid,
        "array_task_id": seed_id - 1,
        "seed_frame_id": frame_id,
        "alpha_initial": alpha_initial,
        "alpha_final": alpha_final,
        "alpha_trajectory": [float(x) for x in alpha_trajectory],
        "n_evaluations": n_evaluations,
        "return_code": return_code,
        "trajectory_sha256": str(result_sha or ""),
        "wall_seconds": wall_seconds,
        "fell_back_to_ds": bool(result.get("fell_back_to_ds", False)),
        "whitened_distance_final": whitened,
        "atom_types": clean_atom_types,
        "final_coordinates": clean_coords,
    }


def write_ariadne_results_manifest(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    path = ariadne_results_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def write_ariadne_landing_audit(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    data = dict(payload)
    data["schema_version"] = ARIADNE_LANDING_AUDIT_SCHEMA_VERSION
    path = ariadne_landing_audit_path(iter_dir)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError(
                "ARIADNE audit is unreadable: " + str(path)
            ) from exc
        if isinstance(existing, dict) and isinstance(existing.get("maturity"), dict):
            data["maturity"] = dict(existing["maturity"])
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, data)
    return path


def read_ariadne_landing_audit(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
) -> Dict[str, Any]:
    path = ariadne_landing_audit_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("ARIADNE landing audit missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE landing audit unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("ARIADNE landing audit must be a JSON object")
    if _required_int(data.get("schema_version", -1), "ARIADNE landing audit schema_version") != ARIADNE_LANDING_AUDIT_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported ARIADNE landing audit schema")
    iteration = _required_int(data.get("iteration"), "ARIADNE landing audit iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("ARIADNE landing audit iteration mismatch")
    summary = data.get("summary")
    seeds = data.get("seeds")
    if not isinstance(summary, dict):
        raise HandoffManifestError("ARIADNE landing audit summary must be an object")
    if not isinstance(seeds, list):
        raise HandoffManifestError("ARIADNE landing audit seeds must be a list")
    seen = set()
    for rec in seeds:
        if not isinstance(rec, dict):
            raise HandoffManifestError("ARIADNE landing audit seed record must be an object")
        seed_id = _required_int(rec.get("seed_id"), "ARIADNE landing audit seed_id")
        if seed_id < 1:
            raise HandoffManifestError("ARIADNE landing audit seed_id must be >= 1")
        if seed_id in seen:
            raise HandoffManifestError("duplicate ARIADNE landing audit seed_id")
        seen.add(seed_id)
    return data


def read_ariadne_results_manifest(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    require_nonempty: bool = True,
) -> Dict[str, Any]:
    path = ariadne_results_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("ARIADNE results manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("ARIADNE results manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("ARIADNE results manifest must be a JSON object")
    if _required_int(data.get("schema_version", -1), "ARIADNE results schema_version") != ARIADNE_RESULTS_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported ARIADNE results manifest schema")
    iteration = _required_int(data.get("iteration"), "ARIADNE results iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("ARIADNE results manifest iteration mismatch")
    accepted = data.get("accepted")
    rejected = data.get("rejected")
    if not isinstance(accepted, list):
        raise HandoffManifestError("ARIADNE results manifest accepted must be a list")
    if not isinstance(rejected, list):
        raise HandoffManifestError("ARIADNE results manifest rejected must be a list")
    n_accepted = data.get("n_accepted")
    if n_accepted is not None and _required_int(n_accepted, "ARIADNE n_accepted") != len(accepted):
        raise HandoffManifestError("ARIADNE n_accepted does not match accepted length")
    n_rejected = data.get("n_rejected")
    if n_rejected is not None and _required_int(n_rejected, "ARIADNE n_rejected") != len(rejected):
        raise HandoffManifestError("ARIADNE n_rejected does not match rejected length")
    expected_n = data.get("expected_n")
    if expected_n is not None and _required_int(expected_n, "ARIADNE expected_n") != len(accepted) + len(rejected):
        raise HandoffManifestError("ARIADNE expected_n does not match accepted+rejected length")
    if require_nonempty and not accepted:
        raise HandoffManifestError("ARIADNE results manifest accepted list is empty")
    from .ariadne_outputs import (
        SEED_OUTPUT_MANIFEST_FILENAME,
        SEED_RESULT_FILENAME,
        validate_seed_output,
    )
    from .seed_identity import read_ariadne_task_map
    from .versioning.manifest import sha256_file
    from .versioning.provenance import PROVENANCE_FILENAME, validate_provenance

    task_map = read_ariadne_task_map(
        Path(iter_dir),
        expected_iteration=iteration,
    )
    task_map_path = ariadne_task_map_path(iter_dir)
    task_map_binding = data.get("task_map")
    if not isinstance(task_map_binding, dict):
        raise HandoffManifestError("ARIADNE results task-map binding is missing")
    bound_task_map = resolve_handoff_path(
        path.parent,
        task_map_binding.get("path", ""),
        kind="ARIADNE results task map",
    )
    if bound_task_map != task_map_path.resolve():
        raise HandoffManifestError("ARIADNE results task-map path mismatch")
    if str(task_map_binding.get("sha256") or "") != sha256_file(bound_task_map):
        raise HandoffManifestError("ARIADNE results task-map hash mismatch")
    if str(data.get("campaign_uid") or "") != str(
        task_map.get("campaign_uid") or ""
    ):
        raise HandoffManifestError("ARIADNE results campaign UID mismatch")
    tasks_by_seed_id = {
        int(task["seed_id"]): dict(task) for task in task_map["tasks"]
    }
    seeds_root = ariadne_seeds_dir(iter_dir)
    expected_seed_names = {
        Path(str(task["seed_directory"])).name for task in task_map["tasks"]
    }
    if seeds_root.exists() or seeds_root.is_symlink():
        if seeds_root.is_symlink() or not seeds_root.is_dir():
            raise HandoffManifestError("ARIADNE seeds root is not a regular directory")
        for child in seeds_root.iterdir():
            if child.is_symlink() or not child.is_dir():
                raise HandoffManifestError(
                    "unexpected non-directory ARIADNE seed entry: " + child.name
                )
            try:
                parse_seed_directory_name(child.name)
            except ValueError as exc:
                raise HandoffManifestError(str(exc)) from exc
            if child.name not in expected_seed_names:
                raise HandoffManifestError(
                    "ARIADNE seed directory is outside TASK_MAP: " + child.name
                )
    if expected_n is None or int(expected_n) != int(task_map["n_tasks"]):
        raise HandoffManifestError("ARIADNE results/task-map count mismatch")
    seen = set()
    normalised_accepted = []
    trajectory_sha = str(data.get("trajectory_sha256") or "")
    for rec in accepted:
        if not isinstance(rec, dict):
            raise HandoffManifestError("accepted ARIADNE record must be an object")
        seed_id = _required_int(rec.get("seed_id"), "accepted ARIADNE seed_id")
        if seed_id not in tasks_by_seed_id:
            raise HandoffManifestError("accepted ARIADNE seed_id is outside TASK_MAP")
        if seed_id in seen:
            raise HandoffManifestError("duplicate accepted ARIADNE seed_id")
        seen.add(seed_id)
        task = tasks_by_seed_id[seed_id]
        if str(rec.get("seed_uid") or "") != str(task["seed_uid"]):
            raise HandoffManifestError("accepted ARIADNE seed_uid mismatch")
        out_rec = dict(rec)
        for key in (
            "seed_dir",
            "result_json",
            "provenance_json",
            "output_manifest",
        ):
            resolved = resolve_handoff_path(
                path.parent,
                rec.get(key, ""),
                kind="ARIADNE " + key,
                directory=(key == "seed_dir"),
            )
            out_rec[key] = str(resolved)
        expected_seed_dir = resolve_handoff_path(
            path.parent,
            task["seed_directory"].removeprefix("ariadne/"),
            kind="ARIADNE task-map seed directory",
            directory=True,
        )
        if Path(out_rec["seed_dir"]) != expected_seed_dir:
            raise HandoffManifestError("ARIADNE seed directory/task-map mismatch")
        expected_paths = {
            "result_json": expected_seed_dir / SEED_RESULT_FILENAME,
            "provenance_json": expected_seed_dir / PROVENANCE_FILENAME,
            "output_manifest": expected_seed_dir / SEED_OUTPUT_MANIFEST_FILENAME,
        }
        for label, expected_path in expected_paths.items():
            if Path(out_rec[label]) != expected_path.resolve():
                raise HandoffManifestError(
                    "ARIADNE " + label + " does not match its canonical seed path"
                )
        output_payload = validate_seed_output(
            expected_seed_dir,
            expected_campaign_uid=str(task_map["campaign_uid"]),
            expected_iteration=iteration,
            expected_seed_id=seed_id,
            expected_seed_uid=str(task["seed_uid"]),
            expected_array_task_id=int(task["array_task_id"]),
        )
        if not bool(output_payload["task_success"]):
            raise HandoffManifestError(
                "accepted ARIADNE seed output records task failure"
            )
        if int(output_payload["task_exit_code"]) != 0:
            raise HandoffManifestError(
                "accepted ARIADNE seed output has non-zero task exit code"
            )
        for field, artefact in (
            ("result_sha256", expected_paths["result_json"]),
            ("output_manifest_sha256", expected_paths["output_manifest"]),
        ):
            if str(out_rec.get(field) or "") != sha256_file(artefact):
                raise HandoffManifestError("accepted ARIADNE " + field + " mismatch")
        validate_provenance(
            expected_seed_dir,
            campaign_uid=str(task_map["campaign_uid"]),
            iteration=iteration,
            trajectory_sha256=trajectory_sha or None,
            seed_frame_id=_int_or_none(out_rec.get("seed_frame_id")),
            seed_id=seed_id,
            seed_uid=str(task["seed_uid"]),
            array_task_id_zero_based=int(task["array_task_id"]),
        )
        try:
            result_payload = json.loads(Path(str(out_rec["result_json"])).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError("ARIADNE result unreadable: " + str(out_rec["result_json"])) from exc
        result_safety = result_payload.get("landing_safety")
        record_safety = out_rec.get("landing_safety")
        if isinstance(result_safety, dict) and isinstance(record_safety, dict):
            if bool(result_safety.get("accepted", False)) != bool(
                record_safety.get("accepted", False)
            ):
                raise HandoffManifestError("landing_safety_mismatch")
        elif isinstance(result_safety, dict):
            out_rec["landing_safety"] = dict(result_safety)
        elif isinstance(record_safety, dict):
            result_payload = dict(result_payload)
            result_payload["landing_safety"] = dict(record_safety)
        seed_record = {
            "seed_id": seed_id,
            "seed_uid": str(task["seed_uid"]),
            "frame_id": _int_or_none(out_rec.get("seed_frame_id")),
        }
        validated = validate_ariadne_result(
            result_payload,
            expected_iteration=iteration,
            seed_record=seed_record,
            expected_trajectory_sha256=trajectory_sha or None,
        )
        normalised_accepted.append(out_rec)
    normalised_rejected = []
    for rec in rejected:
        if not isinstance(rec, dict):
            raise HandoffManifestError("rejected ARIADNE record must be an object")
        out_rec = dict(rec)
        seed_id = _required_int(rec.get("seed_id"), "rejected ARIADNE seed_id")
        if seed_id not in tasks_by_seed_id or seed_id in seen:
            raise HandoffManifestError("invalid or duplicate rejected ARIADNE seed_id")
        seen.add(seed_id)
        task = tasks_by_seed_id[seed_id]
        if str(rec.get("seed_uid") or "") != str(task["seed_uid"]):
            raise HandoffManifestError("rejected ARIADNE seed_uid mismatch")
        if rec.get("array_task_id") is not None and _required_int(
            rec.get("array_task_id"),
            "rejected ARIADNE array_task_id",
        ) != int(task["array_task_id"]):
            raise HandoffManifestError("rejected ARIADNE array_task_id mismatch")
        expected_seed_dir = resolve_handoff_path(
            path.parent,
            str(task["seed_directory"]).removeprefix("ariadne/"),
            kind="ARIADNE rejected task-map seed directory",
            must_exist=False,
            directory=True,
        )
        expected_paths = {
            "seed_dir": expected_seed_dir,
            "result_json": expected_seed_dir / SEED_RESULT_FILENAME,
            "provenance_json": expected_seed_dir / PROVENANCE_FILENAME,
            "output_manifest": expected_seed_dir / SEED_OUTPUT_MANIFEST_FILENAME,
        }
        for key in expected_paths:
            if key in out_rec and out_rec.get(key):
                resolved = _resolve_rejected_handoff_reference(
                    path.parent,
                    out_rec.get(key, ""),
                    kind="ARIADNE rejected " + key,
                    expected=expected_paths[key],
                    directory=(key == "seed_dir"),
                )
                out_rec[key] = str(resolved)
        normalised_rejected.append(out_rec)
    if seen != set(tasks_by_seed_id):
        raise HandoffManifestError("ARIADNE results do not cover every TASK_MAP seed")
    out = dict(data)
    out["accepted"] = normalised_accepted
    out["rejected"] = normalised_rejected
    return out


def ariadne_candidate_frames(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    expected_config_sha256: Optional[str] = None,
    require_batch_decision: bool = True,
) -> Tuple[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    """Return accepted ARIADNE geometries as ICHOR Atoms plus manifest records."""
    from ichor.core.atoms import Atom, Atoms

    if require_batch_decision:
        read_ariadne_batch_decision(
            iter_dir,
            expected_iteration=expected_iteration,
            expected_config_sha256=expected_config_sha256,
            require_accepted=True,
        )
    manifest = read_ariadne_results_manifest(
        iter_dir,
        expected_iteration=expected_iteration,
    )
    frames = []
    records = []
    for rec in manifest["accepted"]:
        result_path = Path(str(rec["result_json"]))
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError("ARIADNE result unreadable: " + str(result_path)) from exc
        result_safety = result.get("landing_safety")
        record_safety = rec.get("landing_safety")
        if isinstance(result_safety, dict) and isinstance(record_safety, dict):
            if bool(result_safety.get("accepted", False)) != bool(
                record_safety.get("accepted", False)
            ):
                raise HandoffManifestError("landing_safety_mismatch")
        elif isinstance(record_safety, dict):
            result = dict(result)
            result["landing_safety"] = dict(record_safety)
        seed_record = {
            "seed_id": int(rec["seed_id"]),
            "seed_uid": str(rec["seed_uid"]),
            "frame_id": _int_or_none(rec.get("seed_frame_id")),
        }
        validate_ariadne_result(
            result,
            expected_iteration=int(manifest["iteration"]),
            seed_record=seed_record,
            expected_trajectory_sha256=str(manifest.get("trajectory_sha256") or "") or None,
        )
        atom_types = result.get("atom_types") or []
        coords = result.get("final_coordinates") or []
        if not atom_types or len(atom_types) != len(coords):
            raise HandoffManifestError("ARIADNE result geometry shape mismatch: " + str(result_path))
        atoms = Atoms([
            Atom(str(t), _finite_float(c[0]), _finite_float(c[1]), _finite_float(c[2]))
            for t, c in zip(atom_types, coords)
        ])
        frames.append(atoms)
        records.append(dict(rec))
    return manifest, frames, records


def authoritative_ariadne_candidate_frames(
    campaign_dir: Any,
    iteration: int,
    *,
    expected_campaign_uid: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    """Return accepted candidates after producer-bound decision validation."""
    from .layout import active_iteration_dir

    campaign = Path(campaign_dir).expanduser().resolve()
    read_authoritative_ariadne_batch_decision(
        campaign,
        int(iteration),
        expected_campaign_uid=expected_campaign_uid,
        require_accepted=True,
    )
    return ariadne_candidate_frames(
        active_iteration_dir(campaign, int(iteration)),
        expected_iteration=int(iteration),
        require_batch_decision=False,
    )


def write_phase_a_sample_manifest(initial_dir: Any, payload: Dict[str, Any]) -> Path:
    path = phase_a_sample_manifest_path(initial_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = PHASE_A_SAMPLE_SCHEMA_VERSION
    selection_root = path.parent.resolve()
    campaign = _phase_a_campaign_root(selection_root)
    root = selection_root.parent
    from .point_allocation import read_point_allocation
    from .versioning.manifest import sha256_file

    sample = resolve_handoff_path(
        root,
        data.get("sample_xyz", ""),
        kind="Phase A sample_xyz",
    )
    index = resolve_handoff_path(
        root,
        data.get("index_path", ""),
        kind="Phase A index_path",
    )
    data["sample_xyz_size"] = int(sample.stat().st_size)
    data["sample_xyz_sha256"] = sha256_file(sample)
    data["index_size"] = int(index.stat().st_size)
    data["index_sha256"] = sha256_file(index)
    allocation_binding = data.get("point_allocation")
    if not isinstance(allocation_binding, dict):
        raise HandoffManifestError("Phase A point_allocation must be an object")
    allocation_path = resolve_handoff_path(
        root,
        allocation_binding.get("manifest", ""),
        kind="Phase A point-allocation manifest",
    )
    supplied_uid = str(data.get("campaign_uid") or "")
    allocation_payload = read_point_allocation(
        allocation_path,
        expected_campaign_uid=supplied_uid or None,
        expected_context="bootstrap",
        expected_iteration=0,
    )
    allocation_uid = str(allocation_payload["campaign_uid"])
    if supplied_uid and supplied_uid != allocation_uid:
        raise HandoffManifestError("Phase A sample manifest campaign UID mismatch")
    data["campaign_uid"] = allocation_uid
    allocation_binding = dict(allocation_binding)
    allocation_binding["slot_assignment_sha256"] = str(
        allocation_payload["slot_assignment_sha256"]
    )
    supplied_targets = allocation_binding.get("targets")
    canonical_targets = dict(allocation_payload["targets"])
    if supplied_targets is not None and dict(supplied_targets) != canonical_targets:
        raise HandoffManifestError("Phase A point-allocation targets mismatch")
    allocation_binding["targets"] = canonical_targets
    data["point_allocation"] = allocation_binding
    from .acquisition.trajectory_pool import (
        POOL_MANIFEST_FILENAME,
        POOL_SUBDIR,
        TrajectoryPool,
    )

    source_pool_raw = data.get("source_pool_manifest")
    if not isinstance(source_pool_raw, str) or not source_pool_raw.strip():
        raise HandoffManifestError(
            "Phase A source_pool_manifest is required"
        )
    source_pool = resolve_handoff_path(
        campaign,
        source_pool_raw,
        kind="Phase A source pool manifest",
    )
    expected_source_pool = (
        campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    ).resolve()
    if source_pool != expected_source_pool:
        raise HandoffManifestError(
            "Phase A source pool manifest is not canonical"
        )
    try:
        pool = TrajectoryPool.load(campaign)
    except (OSError, ValueError) as exc:
        raise HandoffManifestError(
            "Phase A source trajectory pool is invalid: " + str(exc)
        ) from exc
    supplied_trajectory_sha = data.get("trajectory_sha256")
    if (
        supplied_trajectory_sha not in (None, "")
        and str(supplied_trajectory_sha) != str(pool.sha256)
    ):
        raise HandoffManifestError("Phase A trajectory SHA mismatch")
    supplied_pool_frames = data.get("n_pool_frames")
    if (
        supplied_pool_frames is not None
        and _required_int(supplied_pool_frames, "Phase A n_pool_frames")
        != int(pool.n_frames())
    ):
        raise HandoffManifestError("Phase A source pool frame-count mismatch")
    selected = data.get("selected_indices")
    if not isinstance(selected, list):
        raise HandoffManifestError("Phase A selected_indices must be a list")
    selected_pool_indices = [
        _required_int(value, "Phase A selected_indices")
        for value in selected
        if value is not None
    ]
    if any(index >= int(pool.n_frames()) for index in selected_pool_indices):
        raise HandoffManifestError(
            "Phase A selected pool index is outside the source pool"
        )
    supplied_selected_pool = data.get("selected_pool_indices")
    if supplied_selected_pool is not None:
        if not isinstance(supplied_selected_pool, list):
            raise HandoffManifestError(
                "Phase A selected_pool_indices must be a list"
            )
        canonical_supplied = [
            _required_int(value, "Phase A selected_pool_indices")
            for value in supplied_selected_pool
        ]
        if canonical_supplied != selected_pool_indices:
            raise HandoffManifestError(
                "Phase A selected pool indexes disagree with selected_indices"
            )
    data["trajectory_sha256"] = str(pool.sha256)
    data["n_pool_frames"] = int(pool.n_frames())
    data["selected_pool_indices"] = selected_pool_indices
    data["source_pool_manifest_size"] = int(source_pool.stat().st_size)
    data["source_pool_manifest_sha256"] = sha256_file(source_pool)
    custom_bootstrap_raw = data.get("custom_bootstrap_manifest")
    if custom_bootstrap_raw:
        custom_bootstrap = resolve_handoff_path(
            campaign,
            custom_bootstrap_raw,
            kind="Phase A custom-bootstrap manifest",
        )
        data["custom_bootstrap_manifest_size"] = int(
            custom_bootstrap.stat().st_size
        )
        data["custom_bootstrap_manifest_sha256"] = sha256_file(
            custom_bootstrap
        )
    atomic_write_json(path, data)
    return path


def read_phase_a_sample_manifest(
    initial_dir: Any,
    *,
    require_nonempty: bool = True,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    path = phase_a_sample_manifest_path(initial_dir)
    if not path.is_file():
        raise FileNotFoundError("Phase A sample manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("Phase A sample manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("Phase A sample manifest must be a JSON object")
    if _required_int(data.get("schema_version", -1), "Phase A schema_version") != PHASE_A_SAMPLE_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported Phase A sample manifest schema")
    if str(data.get("phase")) != "PHASE_A_DIVERSITY":
        raise HandoffManifestError("Phase A sample manifest phase mismatch")
    if (
        expected_campaign_uid is not None
        and data.get("campaign_uid") not in (None, "")
        and str(data.get("campaign_uid")) != str(expected_campaign_uid)
    ):
        raise HandoffManifestError("Phase A sample manifest campaign UID mismatch")
    if _required_int(data.get("iteration"), "Phase A iteration") != 0:
        raise HandoffManifestError("Phase A sample manifest iteration must be 0")
    from .sampling.diversity_contract import selector_contract_matches

    if not selector_contract_matches(data.get("selector")):
        raise HandoffManifestError("Phase A selector contract is invalid")
    n_select = _required_int(data.get("n_select"), "Phase A n_select")
    if require_nonempty and n_select <= 0:
        raise HandoffManifestError("Phase A sample manifest n_select must be positive")
    selection_root = Path(initial_dir).resolve()
    campaign = _phase_a_campaign_root(selection_root)
    root = selection_root.parent
    sample = resolve_handoff_path(
        root,
        data.get("sample_xyz", ""),
        kind="Phase A sample_xyz",
    )
    if sample.name != "selected.xyz" or sample.parent != selection_root.resolve():
        raise HandoffManifestError("Phase A sample path has unexpected name: " + str(sample))
    index_path = resolve_handoff_path(
        root,
        data.get("index_path", ""),
        kind="Phase A index_path",
    )
    if index_path.name != "selected_indices.dat" or index_path.parent != selection_root.resolve():
        raise HandoffManifestError("Phase A index path has unexpected name: " + str(index_path))
    from .versioning.manifest import sha256_file

    if _required_int(data.get("sample_xyz_size"), "Phase A sample_xyz_size") != int(
        sample.stat().st_size
    ):
        raise HandoffManifestError("Phase A sample XYZ size mismatch")
    if str(data.get("sample_xyz_sha256") or "") != sha256_file(sample):
        raise HandoffManifestError("Phase A sample XYZ hash mismatch")
    if _xyz_frame_count(sample, "Phase A sample XYZ") != n_select:
        raise HandoffManifestError("Phase A sample XYZ frame count mismatch")
    if _required_int(data.get("index_size"), "Phase A index_size") != int(
        index_path.stat().st_size
    ):
        raise HandoffManifestError("Phase A index size mismatch")
    if str(data.get("index_sha256") or "") != sha256_file(index_path):
        raise HandoffManifestError("Phase A index hash mismatch")
    selected = data.get("selected_indices")
    if not isinstance(selected, list):
        raise HandoffManifestError("Phase A selected_indices must be a list")
    if len(selected) != n_select:
        raise HandoffManifestError("Phase A selected_indices length mismatch")
    seen = set()
    for value in selected:
        if value is None:
            continue
        idx = _required_int(value, "Phase A selected_indices")
        if idx in seen:
            raise HandoffManifestError("Phase A selected_indices contains duplicates")
        seen.add(idx)
    expected_index_lines = []
    custom_number = 0
    for value in selected:
        if value is None:
            expected_index_lines.append("custom:" + str(custom_number))
            custom_number += 1
        else:
            expected_index_lines.append(str(int(value)))
    observed_index_lines = index_path.read_text(encoding="utf-8").splitlines()
    if observed_index_lines != expected_index_lines:
        raise HandoffManifestError(
            "Phase A index contents do not match selected_indices"
        )
    allocation = data.get("point_allocation")
    if not isinstance(allocation, dict):
        raise HandoffManifestError("Phase A point_allocation must be an object")
    allocation_manifest = resolve_handoff_path(
        root,
        allocation.get("manifest", ""),
        kind="Phase A point-allocation manifest",
    )
    from .point_allocation import read_point_allocation

    allocation_payload = read_point_allocation(
        allocation_manifest,
        expected_campaign_uid=expected_campaign_uid,
        expected_context="bootstrap",
        expected_iteration=0,
    )
    if str(data.get("campaign_uid") or allocation_payload["campaign_uid"]) != str(
        allocation_payload["campaign_uid"]
    ):
        raise HandoffManifestError("Phase A sample/allocation campaign UID mismatch")
    if str(allocation.get("slot_assignment_sha256") or "") != str(
        allocation_payload.get("slot_assignment_sha256") or ""
    ):
        raise HandoffManifestError("Phase A point-allocation assignment mismatch")
    if str(allocation_payload.get("context") or "") != "bootstrap" or int(
        allocation_payload.get("iteration", -1)
    ) != 0:
        raise HandoffManifestError("Phase A point-allocation context mismatch")
    primary = allocation.get("primary")
    if not isinstance(primary, list) or len(primary) != n_select:
        raise HandoffManifestError("Phase A point-allocation primary count mismatch")
    expected_primary = {
        str(slot["attempts"][0]["candidate_id"]): (
            int(slot["slot_id"]),
            str(slot["split"]),
        )
        for slot in allocation_payload["slots"]
    }
    try:
        observed_primary = {
            str(record["candidate_id"]): (
                int(record["slot_id"]),
                str(record["split"]),
            )
            for record in primary
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HandoffManifestError(
            "Phase A point-allocation primary records are invalid"
        ) from exc
    if len(observed_primary) != len(primary) or observed_primary != expected_primary:
        raise HandoffManifestError(
            "Phase A point-allocation primary records do not match allocation slots"
        )
    if isinstance(selected, list):
        for row_number, (selected_frame, allocation_record) in enumerate(
            zip(selected, primary),
            start=1,
        ):
            allocated_frame = _int_or_none(allocation_record.get("frame_id"))
            if allocated_frame != selected_frame:
                raise HandoffManifestError(
                    "Phase A sample/allocation frame mismatch at row "
                    + str(row_number)
                )
    if dict(allocation.get("targets") or {}) != dict(
        allocation_payload.get("targets") or {}
    ):
        raise HandoffManifestError("Phase A point-allocation targets mismatch")
    allocation = dict(allocation)
    allocation["manifest"] = str(allocation_manifest)
    from .acquisition.trajectory_pool import (
        POOL_MANIFEST_FILENAME,
        POOL_SUBDIR,
        TrajectoryPool,
    )

    source_pool_raw = data.get("source_pool_manifest")
    if not isinstance(source_pool_raw, str) or not source_pool_raw.strip():
        raise HandoffManifestError(
            "Phase A source_pool_manifest is required"
        )
    source_pool = resolve_handoff_path(
        campaign,
        source_pool_raw,
        kind="Phase A source pool manifest",
    )
    if source_pool != (campaign / POOL_SUBDIR / POOL_MANIFEST_FILENAME).resolve():
        raise HandoffManifestError(
            "Phase A source pool manifest is not canonical"
        )
    if _required_int(
        data.get("source_pool_manifest_size"),
        "Phase A source_pool_manifest_size",
    ) != int(source_pool.stat().st_size):
        raise HandoffManifestError("Phase A source pool size mismatch")
    if str(data.get("source_pool_manifest_sha256") or "") != sha256_file(
        source_pool
    ):
        raise HandoffManifestError("Phase A source pool hash mismatch")
    try:
        pool = TrajectoryPool.load(campaign)
    except (OSError, ValueError) as exc:
        raise HandoffManifestError(
            "Phase A source trajectory pool is invalid: " + str(exc)
        ) from exc
    if str(data.get("trajectory_sha256") or "") != str(pool.sha256):
        raise HandoffManifestError("Phase A trajectory SHA mismatch")
    if _required_int(
        data.get("n_pool_frames"), "Phase A n_pool_frames"
    ) != int(pool.n_frames()):
        raise HandoffManifestError("Phase A source pool frame-count mismatch")
    selected_pool_indices = data.get("selected_pool_indices")
    if not isinstance(selected_pool_indices, list):
        raise HandoffManifestError(
            "Phase A selected_pool_indices must be a list"
        )
    canonical_selected_pool = [
        _required_int(value, "Phase A selected_pool_indices")
        for value in selected_pool_indices
    ]
    expected_selected_pool = [
        int(value) for value in selected if value is not None
    ]
    if canonical_selected_pool != expected_selected_pool:
        raise HandoffManifestError(
            "Phase A selected pool indexes disagree with selected_indices"
        )
    if any(index >= int(pool.n_frames()) for index in canonical_selected_pool):
        raise HandoffManifestError(
            "Phase A selected pool index is outside the source pool"
        )
    custom_bootstrap_raw = data.get("custom_bootstrap_manifest")
    if custom_bootstrap_raw:
        custom_bootstrap = resolve_handoff_path(
            campaign,
            custom_bootstrap_raw,
            kind="Phase A custom-bootstrap manifest",
        )
        if _required_int(
            data.get("custom_bootstrap_manifest_size"),
            "Phase A custom_bootstrap_manifest_size",
        ) != int(custom_bootstrap.stat().st_size):
            raise HandoffManifestError(
                "Phase A custom-bootstrap manifest size mismatch"
            )
        if str(
            data.get("custom_bootstrap_manifest_sha256") or ""
        ) != sha256_file(custom_bootstrap):
            raise HandoffManifestError(
                "Phase A custom-bootstrap manifest hash mismatch"
            )
    out = dict(data)
    out["campaign_uid"] = str(allocation_payload["campaign_uid"])
    out["n_select"] = n_select
    out["sample_xyz"] = str(sample)
    out["index_path"] = str(index_path)
    out["point_allocation"] = allocation
    return out


def write_phase_b_selection_manifest(iter_dir: Any, payload: Dict[str, Any]) -> Path:
    path = phase_b_selection_path(iter_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def read_phase_b_selection_manifest(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    require_nonempty: bool = True,
    expected_campaign_uid: Optional[str] = None,
) -> Dict[str, Any]:
    path = phase_b_selection_path(iter_dir)
    if not path.is_file():
        raise FileNotFoundError("Phase B selection manifest missing: " + str(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("Phase B selection manifest unreadable: " + str(path)) from exc
    if not isinstance(data, dict):
        raise HandoffManifestError("Phase B selection manifest must be a JSON object")
    if _required_int(data.get("schema_version", -1), "Phase B schema_version") != PHASE_B_SELECTION_SCHEMA_VERSION:
        raise HandoffManifestError("unsupported Phase B selection manifest schema")
    iteration = _required_int(data.get("iteration"), "Phase B iteration")
    if expected_iteration is not None and iteration != int(expected_iteration):
        raise HandoffManifestError("Phase B selection manifest iteration mismatch")
    if expected_campaign_uid is not None and str(data.get("campaign_uid") or "") != str(
        expected_campaign_uid
    ):
        raise HandoffManifestError("Phase B selection manifest campaign UID mismatch")
    if str(data.get("status") or "") != "complete":
        raise HandoffManifestError(
            "Phase B selection did not complete: "
            + str(data.get("failure_reason") or "unknown failure")
        )
    root = Path(iter_dir)
    from .versioning.manifest import sha256_file
    considered = data.get("considered")
    final = data.get("final")
    if not isinstance(considered, list):
        raise HandoffManifestError(
            "Phase B selection manifest considered must be a list"
        )
    if not isinstance(final, list):
        raise HandoffManifestError("Phase B selection manifest final must be a list")
    if data.get("n_considered") is not None and _required_int(
        data.get("n_considered"), "Phase B n_considered"
    ) != len(considered):
        raise HandoffManifestError(
            "Phase B n_considered does not match considered length"
        )
    if data.get("n_kept") is not None and _required_int(data.get("n_kept"), "Phase B n_kept") != len(final):
        raise HandoffManifestError("Phase B n_kept does not match final length")
    source_manifest_raw = data.get("source_ariadne_manifest")
    if not isinstance(source_manifest_raw, str) or not source_manifest_raw:
        raise HandoffManifestError("Phase B source ARIADNE manifest is missing")
    source_manifest = resolve_handoff_path(
        root,
        source_manifest_raw,
        kind="Phase B source_ariadne_manifest",
    )
    if source_manifest != ariadne_results_path(iter_dir).resolve():
        raise HandoffManifestError("Phase B source ARIADNE manifest is noncanonical")
    if sha256_file(source_manifest) != str(
        data.get("source_ariadne_manifest_sha256") or ""
    ):
        raise HandoffManifestError("Phase B source ARIADNE manifest hash mismatch")
    read_ariadne_batch_decision(
        iter_dir,
        expected_iteration=iteration,
        require_accepted=True,
    )
    source_results = read_ariadne_results_manifest(
        iter_dir,
        expected_iteration=iteration,
        require_nonempty=require_nonempty,
    )
    source_by_seed = {
        int(record["seed_id"]): dict(record) for record in source_results["accepted"]
    }

    def verify_source_record(record: Dict[str, Any], *, label: str) -> None:
        seed_id = _required_int(record.get("seed_id"), label + " seed_id")
        source = source_by_seed.get(seed_id)
        if source is None:
            raise HandoffManifestError(label + " seed is outside accepted ARIADNE results")
        if str(record.get("seed_uid") or "") != str(source.get("seed_uid") or ""):
            raise HandoffManifestError(label + " seed UID mismatch")
        for key in ("seed_dir", "result_json", "provenance_json", "output_manifest"):
            if Path(str(record[key])).resolve() != Path(str(source[key])).resolve():
                raise HandoffManifestError(label + " " + key + " source mismatch")
        for key in ("result_sha256", "output_manifest_sha256"):
            if str(record.get(key) or "") != str(source.get(key) or ""):
                raise HandoffManifestError(label + " " + key + " source mismatch")
        safety = source.get("landing_safety")
        if not isinstance(safety, dict) or safety.get("accepted") is not True:
            raise HandoffManifestError(label + " source landing safety is not accepted")
    if require_nonempty and not final:
        raise HandoffManifestError("Phase B selection manifest final list is empty")
    allocation = data.get("point_allocation")
    if not isinstance(allocation, dict):
        raise HandoffManifestError("Phase B point_allocation must be an object")
    allocation_manifest = resolve_handoff_path(
        root,
        allocation.get("manifest", ""),
        kind="Phase B point-allocation manifest",
    )
    targets = allocation.get("targets")
    reserve = allocation.get("reserve")
    if not isinstance(targets, dict) or not isinstance(reserve, list):
        raise HandoffManifestError("Phase B point-allocation targets/reserve are invalid")
    from .point_allocation import read_point_allocation

    allocation_payload = read_point_allocation(
        allocation_manifest,
        expected_campaign_uid=expected_campaign_uid,
        expected_context="active",
        expected_iteration=iteration,
    )
    if str(allocation.get("slot_assignment_sha256") or "") != str(
        allocation_payload.get("slot_assignment_sha256") or ""
    ):
        raise HandoffManifestError("Phase B point-allocation assignment mismatch")
    from .sampling.diversity_contract import selector_contract_matches

    if not selector_contract_matches(data.get("selector")):
        raise HandoffManifestError("Phase B selector contract is invalid")
    normalised_bindings = {}
    for label in ("considered_candidates_xyz", "selected_xyz"):
        binding = data.get(label)
        if not isinstance(binding, dict):
            raise HandoffManifestError("Phase B " + label + " binding is missing")
        bound = resolve_handoff_path(root, binding.get("path", ""), kind="Phase B " + label)
        if int(binding.get("size", -1)) != int(bound.stat().st_size):
            raise HandoffManifestError("Phase B " + label + " size mismatch")
        if str(binding.get("sha256") or "") != sha256_file(bound):
            raise HandoffManifestError("Phase B " + label + " hash mismatch")
        normalised_bindings[label] = {**binding, "path": str(bound)}
    protocol = data.get("sampling_protocol")
    if not isinstance(protocol, dict):
        raise HandoffManifestError("Phase B sampling_protocol binding is missing")
    normalised_protocol = dict(protocol)
    for label in ("resolved", "audit", "scale_model"):
        binding = protocol.get(label)
        if not isinstance(binding, dict):
            raise HandoffManifestError("Phase B protocol " + label + " binding is missing")
        bound = resolve_handoff_path(
            root,
            binding.get("path", ""),
            kind="Phase B protocol " + label,
        )
        if int(binding.get("size", -1)) != int(bound.stat().st_size):
            raise HandoffManifestError("Phase B protocol " + label + " size mismatch")
        if str(binding.get("sha256") or "") != sha256_file(bound):
            raise HandoffManifestError("Phase B protocol " + label + " hash mismatch")
        normalised_protocol[label] = {**binding, "path": str(bound)}
    seen_considered = set()
    normalised_considered = []
    considered_kept_final_indexes = set()
    for considered_idx, rec in enumerate(considered):
        if not isinstance(rec, dict):
            raise HandoffManifestError("Phase B considered record must be an object")
        out_rec = dict(rec)
        declared_considered_rank = _required_int(
            out_rec.get("considered_rank"), "Phase B considered_rank"
        )
        if declared_considered_rank != considered_idx + 1:
            raise HandoffManifestError(
                "Phase B considered_rank values must be contiguous from one"
            )
        if declared_considered_rank in seen_considered:
            raise HandoffManifestError("duplicate Phase B considered_rank")
        seen_considered.add(declared_considered_rank)
        kept_value = out_rec.get("kept_after_dedup")
        if kept_value is not True and kept_value is not False:
            raise HandoffManifestError(
                "Phase B kept_after_dedup must be a JSON Boolean"
            )
        kept = kept_value is True
        final_rank_value = out_rec.get("final_rank")
        if kept:
            if final_rank_value is None:
                raise HandoffManifestError(
                    "Phase B kept considered record has null final_rank"
                )
            considered_kept_final_indexes.add(
                _required_int(final_rank_value, "Phase B considered final_rank")
            )
        elif final_rank_value is not None:
            raise HandoffManifestError(
                "Phase B dropped considered record has non-null final_rank"
            )
        for key in ("seed_dir", "result_json", "provenance_json", "output_manifest"):
            resolved = resolve_handoff_path(
                root,
                out_rec.get(key, ""),
                kind="Phase B considered " + key,
                directory=(key == "seed_dir"),
            )
            out_rec[key] = str(resolved)
        if str(out_rec.get("provenance_sha256") or "") != sha256_file(
            Path(out_rec["provenance_json"])
        ):
            raise HandoffManifestError("Phase B considered provenance hash mismatch")
        verify_source_record(out_rec, label="Phase B considered")
        normalised_considered.append(out_rec)
    seen_final = set()
    seen_final_considered = set()
    normalised_final = []
    for rec in final:
        if not isinstance(rec, dict):
            raise HandoffManifestError("Phase B final record must be an object")
        out_rec = dict(rec)
        final_rank = _required_int(rec.get("final_rank"), "Phase B final_rank")
        if final_rank in seen_final:
            raise HandoffManifestError("duplicate Phase B final_rank")
        seen_final.add(final_rank)
        considered_rank = _required_int(
            out_rec.get("considered_rank"), "Phase B final considered_rank"
        )
        if considered_rank in seen_final_considered:
            raise HandoffManifestError("duplicate Phase B final considered_rank")
        seen_final_considered.add(considered_rank)
        if out_rec.get("kept_after_dedup") is not True:
            raise HandoffManifestError("Phase B final record is not marked kept_after_dedup")
        for key in ("seed_dir", "result_json", "provenance_json", "output_manifest"):
            resolved = resolve_handoff_path(
                root,
                rec.get(key, ""),
                kind="Phase B " + key,
                directory=(key == "seed_dir"),
            )
            out_rec[key] = str(resolved)
        if str(out_rec.get("provenance_sha256") or "") != sha256_file(
            Path(out_rec["provenance_json"])
        ):
            raise HandoffManifestError("Phase B final provenance hash mismatch")
        verify_source_record(out_rec, label="Phase B final")
        normalised_final.append(out_rec)
    if seen_final and sorted(seen_final) != list(range(1, len(seen_final) + 1)):
        raise HandoffManifestError("Phase B final_rank values must be contiguous from one")
    if seen_final != considered_kept_final_indexes:
        raise HandoffManifestError(
            "Phase B final records do not match kept considered records"
        )
    kept_considered_indexes = {
        _required_int(rec.get("considered_rank"), "Phase B kept considered_rank")
        for rec in normalised_considered
        if rec.get("kept_after_dedup") is True
    }
    if seen_final_considered != kept_considered_indexes:
        raise HandoffManifestError(
            "Phase B final considered_rank set does not match kept considered records"
        )
    out = dict(data)
    out["considered"] = normalised_considered
    out["final"] = normalised_final
    normalised_allocation = dict(allocation)
    normalised_allocation["manifest"] = str(allocation_manifest)
    out["point_allocation"] = normalised_allocation
    out.update(normalised_bindings)
    out["sampling_protocol"] = normalised_protocol
    if source_manifest is not None:
        out["source_ariadne_manifest"] = str(source_manifest)
    return out


def _read_xyz_geometry_records(path: Path) -> List[Dict[str, Any]]:
    from ichor.core.files.xyz.strict_xyz import read_xyz_frames

    try:
        frames = read_xyz_frames(path)
    except (OSError, ValueError) as exc:
        raise HandoffManifestError("Phase B selected XYZ is unreadable: " + str(path)) from exc
    if not frames or any(len(frame) <= 0 for frame in frames):
        raise HandoffManifestError("Phase B selected XYZ contains no frames")
    return [
        {
            "atom_types": [str(atom.type) for atom in frame],
            "coordinates": [
                [float(atom.x), float(atom.y), float(atom.z)] for atom in frame
            ],
        }
        for frame in frames
    ]


def validate_phase_b_handoff(
    iter_dir: Any,
    *,
    expected_iteration: Optional[int] = None,
    expected_campaign_uid: Optional[str] = None,
    coordinate_tolerance: float = 5.0e-6,
) -> Dict[str, Any]:
    """Validate the complete Phase B geometry and provenance handoff."""
    from .versioning.provenance import validate_provenance

    manifest = read_phase_b_selection_manifest(
        iter_dir,
        expected_iteration=expected_iteration,
        expected_campaign_uid=expected_campaign_uid,
        require_nonempty=True,
    )
    campaign_uid = str(manifest.get("campaign_uid") or "")
    if not campaign_uid:
        raise HandoffManifestError("Phase B campaign_uid is missing")
    if expected_campaign_uid is not None and campaign_uid != str(expected_campaign_uid):
        raise HandoffManifestError("Phase B campaign UID mismatch")
    iteration = int(manifest["iteration"])
    frames = _read_xyz_geometry_records(Path(str(manifest["selected_xyz"]["path"])))
    final_records = list(manifest.get("final") or [])
    if len(frames) != len(final_records):
        raise HandoffManifestError("Phase B selected XYZ/final record count mismatch")
    tolerance = float(coordinate_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise HandoffManifestError("Phase B coordinate tolerance is invalid")
    allocation = dict(manifest.get("point_allocation") or {})
    assignment_sha = str(allocation.get("slot_assignment_sha256") or "")
    for index, (frame, record) in enumerate(zip(frames, final_records)):
        result_path = Path(str(record.get("result_json") or ""))
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffManifestError(
                "Phase B selected result is unreadable at index " + str(index)
            ) from exc
        atom_types = [str(value) for value in (result.get("atom_types") or [])]
        coordinates = result.get("final_coordinates") or []
        if atom_types != frame["atom_types"] or len(coordinates) != len(atom_types):
            raise HandoffManifestError(
                "Phase B selected geometry atom ordering mismatch at index " + str(index)
            )
        for atom_index, (observed, expected) in enumerate(
            zip(frame["coordinates"], coordinates)
        ):
            if not isinstance(expected, list) or len(expected) != 3:
                raise HandoffManifestError("Phase B result coordinate shape is invalid")
            for axis, (observed_value, expected_value) in enumerate(zip(observed, expected)):
                expected_float = float(_finite_float(expected_value))
                if abs(float(observed_value) - expected_float) > tolerance:
                    raise HandoffManifestError(
                        "Phase B selected geometry differs from ARIADNE result at frame "
                        + str(index)
                        + " atom "
                        + str(atom_index)
                        + " axis "
                        + str(axis)
                    )
        validate_provenance(
            Path(str(record["seed_dir"])),
            campaign_uid=campaign_uid,
            iteration=iteration,
            seed_frame_id=_int_or_none(record.get("seed_frame_id")),
            seed_id=_required_int(record.get("seed_id"), "Phase B seed_id"),
            seed_uid=str(record.get("seed_uid") or ""),
            array_task_id_zero_based=_required_int(
                record.get("array_task_id"),
                "Phase B array_task_id",
            ),
            require_phase_b_selected=True,
            allocation_split=str(record.get("split") or ""),
            allocation_slot_id=_required_int(record.get("slot_id"), "Phase B slot_id"),
            allocation_candidate_id=str(record.get("candidate_id") or ""),
            allocation_context="active",
            allocation_slot_assignment_sha256=assignment_sha,
        )
    return manifest
