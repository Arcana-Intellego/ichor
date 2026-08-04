"""Validated reuse of expensive ARIADNE resource evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from ..acquisition.trajectory_pool import POOL_XYZ_FILENAME
from ..execution_identity import (
    read_active_environment_generation,
    read_environment_generation,
)
from ..handoff_manifests import ariadne_task_map_path
from ..layout import active_iteration_dir
from ..seed_identity import read_ariadne_task_map
from ..versioning.manifest import sha256_file
from ..versioning.trained_models import resolve_trained_model_set
from .config_lock import read_historical_config_by_fingerprint
from .environment_equivalence import (
    SCIENTIFIC_FINGERPRINT_ALGORITHM,
    assess_resource_evidence_code_equivalence,
)
from .filesystem import campaign_owned_path
from .resource_records import (
    read_resolution,
    resolution_path,
    verify_scientific_evidence,
)
from .resource_solver import AriadneResourceAuthorityContext
from .submission_intent import intent_attempt_records


@dataclass(frozen=True)
class ReusableAriadneResourceEvidence:
    evidence: Dict[str, Any]
    source_submission_identity: str
    source_attempt_id: str
    source_resolution_path: str
    source_resolution_sha256: str
    source_task_count: int
    already_filtered: bool
    fingerprint_algorithm: str = SCIENTIFIC_FINGERPRINT_ALGORITHM
    equivalence_basis: str = "same_environment_generation"


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(label + " must be an integer")
    parsed = int(value)
    if parsed < int(minimum):
        raise ValueError(label + " must be >= " + str(int(minimum)))
    return parsed


def _dimension_contract(config: Any) -> Dict[str, Any]:
    payload = config.to_dict()
    acquisition = payload.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise ValueError("campaign acquisition configuration is missing")
    gradient = acquisition.get("gradient")
    subspace = acquisition.get("subspace")
    if not isinstance(gradient, Mapping) or not isinstance(subspace, Mapping):
        raise ValueError("campaign ARIADNE subspace configuration is missing")
    return {
        "gradient_mode": gradient.get("mode"),
        "subspace": dict(subspace),
    }


def _file_identity(
    evidence: Mapping[str, Any],
    field: str,
) -> Tuple[str, int, str]:
    value = evidence.get(field)
    if not isinstance(value, Mapping):
        raise ValueError("ARIADNE resource evidence has no " + field + " record")
    path = value.get("path")
    size = value.get("size")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise ValueError("ARIADNE resource evidence " + field + " record is invalid")
    return path, int(size), digest


def _candidate_resolution_path(
    campaign: Path,
    intent: Mapping[str, Any],
    *,
    allow_unbound: bool,
) -> Optional[Path]:
    raw_path = intent.get("resource_resolution_path")
    if isinstance(raw_path, str) and raw_path:
        path = campaign_owned_path(campaign, Path(raw_path))
    elif allow_unbound:
        path = resolution_path(
            campaign,
            str(intent["phase"]),
            int(intent["iteration"]),
            str(intent["submission_identity"]),
        )
    else:
        return None
    canonical = resolution_path(
        campaign,
        str(intent["phase"]),
        int(intent["iteration"]),
        str(intent["submission_identity"]),
    )
    if path.resolve() != canonical.resolve():
        raise ValueError(
            "submission intent resource-resolution path is not canonical"
        )
    return path


def _resolution_for_candidate(
    campaign: Path,
    intent: Mapping[str, Any],
    *,
    allow_unbound: bool,
) -> Optional[Tuple[Path, str, Dict[str, Any]]]:
    raw_path = intent.get("resource_resolution_path")
    expected_sha = intent.get("resource_resolution_sha256")
    bound = isinstance(raw_path, str) and bool(raw_path)
    path = _candidate_resolution_path(
        campaign,
        intent,
        allow_unbound=allow_unbound,
    )
    if path is None:
        return None
    if bound:
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(
                "submission intent resource-resolution digest is invalid"
            )
    else:
        expected_sha = None
    if not path.exists() and not path.is_symlink():
        if bound and allow_unbound:
            raise ValueError("active bound resource resolution is missing")
        return None
    if path.is_symlink() or not path.is_file():
        raise ValueError("resource resolution is not a regular file: " + str(path))
    observed_sha = sha256_file(path)
    if expected_sha is not None and observed_sha != str(expected_sha):
        raise ValueError("resource-resolution digest contradicts its intent")
    payload = read_resolution(path)
    expected_identity = (
        str(intent["campaign_uid"]),
        str(intent["phase"]),
        int(intent["iteration"]),
        str(intent["attempt_id"]),
        str(intent["submission_identity"]),
    )
    observed_identity = (
        str(payload.get("campaign_uid") or ""),
        str(payload.get("phase") or ""),
        int(payload.get("iteration", -1)),
        str(payload.get("attempt_id") or ""),
        str(payload.get("submission_identity") or ""),
    )
    if observed_identity != expected_identity:
        raise ValueError("resource-resolution identity contradicts its intent")
    verify_scientific_evidence(campaign, payload["evidence"])
    return path, observed_sha, payload


def _validated_dimensions(
    evidence: Mapping[str, Any],
    *,
    logical_total: int,
    submitted_task_ids: Sequence[int],
) -> Tuple[Tuple[int, ...], bool, bool]:
    raw_dimensions = evidence.get("gradient_dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ValueError("ARIADNE resource evidence has no gradient dimensions")
    dimensions = []
    for raw in raw_dimensions:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError("ARIADNE resource evidence dimension is invalid")
        dimensions.append(int(raw))
    raw_maximum = evidence.get("gradient_dimension")
    if (
        isinstance(raw_maximum, bool)
        or not isinstance(raw_maximum, int)
        or int(raw_maximum) != max(dimensions)
    ):
        raise ValueError("ARIADNE resource evidence maximum dimension is invalid")
    submitted = tuple(int(value) for value in submitted_task_ids)
    recorded_ids = evidence.get("submitted_logical_task_ids")
    if recorded_ids is None:
        if _exact_int(
            evidence.get("n_tasks"),
            "ARIADNE resource evidence n_tasks",
        ) != int(logical_total):
            raise ValueError("ARIADNE full resource evidence task count is invalid")
        if len(dimensions) != int(logical_total):
            raise ValueError(
                "ARIADNE full resource evidence dimensions are incomplete"
            )
        selected = tuple(dimensions[task_id] for task_id in submitted)
        full_submission = submitted == tuple(range(int(logical_total)))
        return selected, full_submission, True
    if not isinstance(recorded_ids, list):
        raise ValueError("ARIADNE filtered resource task identities are invalid")
    parsed_ids = tuple(
        _exact_int(
            value,
            "ARIADNE filtered resource logical task identity",
        )
        for value in recorded_ids
    )
    if (
        len(set(parsed_ids)) != len(parsed_ids)
        or _exact_int(
            evidence.get("logical_n_tasks"),
            "ARIADNE filtered resource logical_n_tasks",
        )
        != int(logical_total)
        or _exact_int(
            evidence.get("n_tasks"),
            "ARIADNE filtered resource n_tasks",
        )
        != len(parsed_ids)
        or len(dimensions) != len(parsed_ids)
    ):
        raise ValueError("ARIADNE filtered resource evidence is inconsistent")
    full_coverage = parsed_ids == tuple(range(int(logical_total)))
    if not full_coverage and parsed_ids != submitted:
        raise ValueError(
            "ARIADNE filtered resource evidence does not cover this retry"
        )
    if full_coverage:
        selected = tuple(dimensions[task_id] for task_id in submitted)
        return selected, submitted == parsed_ids, True
    return tuple(dimensions), True, False


def _candidate_is_compatible(
    campaign: Path,
    current_config: Any,
    current_generation: Mapping[str, Any],
    intent: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    expected_scheduler_kind: str,
    expected_models_version: int,
    task_map: Mapping[str, Any],
    task_map_file: Path,
    logical_total: int,
    submitted_task_ids: Sequence[int],
    authority_context: Optional[AriadneResourceAuthorityContext] = None,
) -> Tuple[Tuple[int, ...], bool, bool, str]:
    if str(intent.get("scheduler_identity_kind") or "slurm").lower() != str(
        expected_scheduler_kind
    ).lower():
        raise ValueError("scheduler kind changed")
    resources = payload.get("resources")
    evidence = payload.get("evidence")
    if not isinstance(resources, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("resource resolution payload is incomplete")
    if str(resources.get("backend") or "").lower() != "ariadne":
        raise ValueError("resource resolution backend is not ARIADNE")
    task_path, _task_size, task_sha = _file_identity(evidence, "task_map")
    if Path(task_path).resolve() != task_map_file.resolve():
        raise ValueError("ARIADNE resource evidence task-map path changed")
    expected_task_sha = (
        str(authority_context.task_map_evidence["sha256"])
        if authority_context is not None
        else sha256_file(task_map_file)
    )
    if task_sha != expected_task_sha:
        raise ValueError("ARIADNE resource evidence task-map digest changed")
    pool_path, _pool_size, pool_sha = _file_identity(evidence, "pool")
    canonical_pool = campaign_owned_path(campaign, campaign / POOL_XYZ_FILENAME)
    if Path(pool_path).resolve() != canonical_pool.resolve():
        raise ValueError("ARIADNE resource evidence pool path changed")
    if pool_sha != str(task_map["trajectory_sha256"]):
        raise ValueError("ARIADNE resource evidence pool digest changed")
    if _exact_int(
        evidence.get("models_version"),
        "ARIADNE resource evidence models_version",
    ) != int(expected_models_version):
        raise ValueError("ARIADNE resource evidence model version changed")
    model_set = (
        authority_context.model_set
        if authority_context is not None
        else resolve_trained_model_set(
            campaign,
            int(expected_models_version),
            verification="metadata",
        )
    )
    if (
        str(evidence.get("model_manifest_sha256") or "")
        != str(model_set.head_manifest_sha256)
        or str(evidence.get("model_set_sha256") or "")
        != str(model_set.model_set_sha256)
    ):
        raise ValueError("ARIADNE resource evidence model identity changed")
    producer_generation = read_environment_generation(
        campaign,
        generation=int(intent["environment_generation"]),
        expected_campaign_uid=str(intent["campaign_uid"]),
    )
    if (
        str(producer_generation.get("digest_sha256") or "")
        != str(intent["environment_generation_digest_sha256"])
    ):
        raise ValueError("producer environment generation digest changed")
    source_config = read_historical_config_by_fingerprint(
        campaign,
        str(producer_generation["campaign_config_sha256"]),
        expected_campaign_uid=str(intent["campaign_uid"]),
    )
    if _dimension_contract(source_config) != _dimension_contract(current_config):
        raise ValueError("ARIADNE resource-evidence configuration changed")
    equivalence_basis = "same_environment_generation"
    if (
        str(producer_generation.get("digest_sha256"))
        != str(current_generation.get("digest_sha256"))
    ):
        code = assess_resource_evidence_code_equivalence(
            producer_generation,
            current_generation,
            backend="ariadne",
        )
        if not bool(code["equivalent"]):
            raise ValueError("ARIADNE resource-evidence producer code changed")
        equivalence_basis = str(code["fingerprint_algorithm"])
    dimensions, already_filtered, full_coverage = _validated_dimensions(
        evidence,
        logical_total=int(logical_total),
        submitted_task_ids=submitted_task_ids,
    )
    return (
        dimensions,
        already_filtered,
        full_coverage,
        equivalence_basis,
    )


def resolve_reusable_ariadne_resource_evidence(
    campaign_dir: Union[str, Path],
    current_config: Any,
    active_intent: Mapping[str, Any],
    *,
    expected_campaign_uid: str,
    iteration: int,
    replacement_round: int,
    expected_scheduler_kind: str,
    expected_models_version: int,
    submitted_task_ids: Sequence[int],
    authority_context: Optional[AriadneResourceAuthorityContext] = None,
) -> Optional[ReusableAriadneResourceEvidence]:
    """Return proven current-attempt or full producer evidence for one retry."""
    campaign = Path(campaign_dir).expanduser().resolve()
    submitted = tuple(
        _exact_int(value, "ARIADNE submitted logical task identity")
        for value in submitted_task_ids
    )
    if not submitted:
        return None
    if len(set(submitted)) != len(submitted):
        raise ValueError("ARIADNE submitted task identities contain duplicates")
    records = list(
        intent_attempt_records(
            campaign,
            "ARIADNE_ARRAY",
            int(iteration),
            expected_campaign_uid=str(expected_campaign_uid),
        )
    )
    ordered = [dict(active_intent)]
    ordered.extend(
        record
        for record in records
        if str(record.get("attempt_id")) != str(active_intent.get("attempt_id"))
    )
    eligible = []
    for index, intent in enumerate(ordered):
        if (
            str(intent.get("campaign_uid") or "") != str(expected_campaign_uid)
            or str(intent.get("phase") or "") != "ARIADNE_ARRAY"
            or int(intent.get("iteration", -1)) != int(iteration)
            or int(intent.get("replacement_round", 0))
            != int(replacement_round)
        ):
            continue
        path = _candidate_resolution_path(
            campaign,
            intent,
            allow_unbound=(index == 0),
        )
        if path is not None and (path.exists() or path.is_symlink()):
            eligible.append((index, intent))
        elif index == 0 and intent.get("resource_resolution_path"):
            raise ValueError("active bound resource resolution is missing")
    if not eligible:
        return None
    if authority_context is not None:
        if authority_context.campaign_dir != campaign:
            raise ValueError("ARIADNE resource authority campaign changed")
        if int(authority_context.iteration) != int(iteration):
            raise ValueError("ARIADNE resource authority iteration changed")
        if str(authority_context.campaign_uid) != str(expected_campaign_uid):
            raise ValueError("ARIADNE resource authority campaign UID changed")
        if int(authority_context.model_set.version) != int(
            expected_models_version
        ):
            raise ValueError("ARIADNE resource authority model version changed")
        authority_context.assert_unchanged(verify_model_payloads=False)
        task_map_file = authority_context.task_map_file
        task_map = authority_context.task_map
    else:
        iter_dir = active_iteration_dir(campaign, int(iteration))
        task_map_file = ariadne_task_map_path(iter_dir)
        task_map = read_ariadne_task_map(
            iter_dir,
            expected_iteration=int(iteration),
        )
    logical_total = int(task_map["n_tasks"])
    if any(value < 0 or value >= logical_total for value in submitted):
        raise ValueError("ARIADNE retry task identity is outside the task map")
    active = read_active_environment_generation(
        campaign,
        expected_campaign_uid=str(expected_campaign_uid),
    )
    current_generation = active["generation"]
    compatible = []
    for index, intent in eligible:
        loaded = _resolution_for_candidate(
            campaign,
            intent,
            allow_unbound=(index == 0),
        )
        if loaded is None:
            continue
        path, digest, payload = loaded
        try:
            (
                dimensions,
                already_filtered,
                full_coverage,
                equivalence_basis,
            ) = _candidate_is_compatible(
                campaign,
                current_config,
                current_generation,
                intent,
                payload,
                expected_scheduler_kind=expected_scheduler_kind,
                expected_models_version=int(expected_models_version),
                task_map=task_map,
                task_map_file=task_map_file,
                logical_total=logical_total,
                submitted_task_ids=submitted,
                authority_context=authority_context,
            )
        except (FileNotFoundError, OSError, ValueError):
            if index == 0:
                raise
            continue
        if index == 0 and not already_filtered:
            raise ValueError(
                "active-attempt ARIADNE resource evidence does not match "
                "the submitted task map"
            )
        compatible.append(
            (
                index == 0,
                full_coverage,
                intent,
                path,
                digest,
                dict(payload["evidence"]),
                dimensions,
                already_filtered,
                equivalence_basis,
            )
        )
    if not compatible:
        return None
    signatures = {item[6] for item in compatible}
    if len(signatures) != 1:
        raise ValueError(
            "compatible ARIADNE resource records contain conflicting dimensions"
        )
    active_candidates = [item for item in compatible if item[0]]
    full_candidates = [item for item in compatible if item[1]]
    if active_candidates:
        selected = active_candidates[0]
    elif full_candidates:
        selected = full_candidates[0]
    else:
        # Historical subset records cannot define the original logical task
        # space for a later retry.
        return None
    (
        _is_active,
        _is_full,
        intent,
        path,
        digest,
        evidence,
        _dimensions,
        already_filtered,
        equivalence_basis,
    ) = selected
    return ReusableAriadneResourceEvidence(
        evidence=evidence,
        source_submission_identity=str(intent["submission_identity"]),
        source_attempt_id=str(intent["attempt_id"]),
        source_resolution_path=str(path),
        source_resolution_sha256=str(digest),
        source_task_count=int(evidence["n_tasks"]),
        already_filtered=bool(already_filtered),
        fingerprint_algorithm=SCIENTIFIC_FINGERPRINT_ALGORITHM,
        equivalence_basis=str(equivalence_basis),
    )


__all__ = [
    "ReusableAriadneResourceEvidence",
    "resolve_reusable_ariadne_resource_evidence",
]
