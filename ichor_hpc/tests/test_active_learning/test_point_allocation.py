"""Exact point-allocation and bounded replacement protocol tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.point_allocation import (
    accepted_attempts,
    allocate_replacements,
    allocation_targets,
    create_point_allocation,
    pending_attempts,
    read_point_allocation,
    record_quantum_results,
    revalidate_rejected_quantum_results,
)


def _candidate(index: int, *, reserve: bool = False):
    record = {
        "candidate_id": "candidate-" + str(index),
        "frame_id": int(index),
    }
    if reserve:
        record["reserve_rank"] = int(index)
    return record


def _create(
    tmp_path: Path,
    *,
    context: str = "active",
    targets=None,
    n_reserve: int = 3,
    mandatory=(),
    forced_splits=None,
):
    if targets is None:
        targets = {"train": 2, "int_val": 1, "ext_val": 0, "total": 3}
    path = tmp_path / "POINT_ALLOCATION.json"
    primary = [_candidate(i) for i in range(int(targets["total"]))]
    reserve = [
        _candidate(int(targets["total"]) + i, reserve=True)
        for i in range(int(n_reserve))
    ]
    payload = create_point_allocation(
        path,
        campaign_uid="campaign-uid",
        context=context,
        iteration=0 if str(context) == "bootstrap" else 1,
        targets=targets,
        primary_candidates=primary,
        reserve_candidates=reserve,
        mandatory_candidate_ids=list(mandatory),
        forced_candidate_splits=dict(forced_splits or {}),
    )
    return path, payload


def _results(attempts, accepted_ids):
    accepted = set(accepted_ids)
    return [
        {
            "candidate_id": attempt["candidate_id"],
            "accepted": attempt["candidate_id"] in accepted,
            "pointdir": "/staging/" + attempt["candidate_id"] + ".pointdir",
            "quality_manifest": (
                "/staging/" + attempt["candidate_id"] + "-quality.json"
                if attempt["candidate_id"] in accepted
                else None
            ),
            "reason": (
                None
                if attempt["candidate_id"] in accepted
                else "synthetic_qm_failure"
            ),
        }
        for attempt in attempts
    ]


def test_config_targets_are_exact_integer_counts():
    config = CampaignConfig()

    assert allocation_targets(config, "bootstrap") == {
        "train": 8,
        "int_val": 2,
        "ext_val": 2,
        "total": 12,
    }
    assert allocation_targets(config, "active") == {
        "train": 3,
        "int_val": 1,
        "ext_val": 0,
        "total": 4,
    }


def test_custom_candidates_are_forced_into_declared_slots(tmp_path):
    targets = {"train": 2, "int_val": 1, "ext_val": 1, "total": 4}
    path, payload = _create(
        tmp_path,
        context="bootstrap",
        targets=targets,
        mandatory=("candidate-0", "candidate-1"),
        forced_splits={"candidate-0": "train", "candidate-1": "train"},
    )

    custom_attempts = [
        attempt
        for slot in payload["slots"]
        for attempt in slot["attempts"]
        if attempt.get("mandatory_custom")
    ]
    assert {attempt["candidate_id"] for attempt in custom_attempts} == {
        "candidate-0",
        "candidate-1",
    }
    assert all(
        slot["split"] == "train"
        for slot in read_point_allocation(path)["slots"]
        if slot["attempts"][0].get("mandatory_custom")
    )


def test_replacement_inherits_failed_slot_and_split(tmp_path):
    path, payload = _create(tmp_path)
    initial = pending_attempts(payload)
    failed = initial[1]
    accepted_ids = {
        attempt["candidate_id"]
        for attempt in initial
        if attempt["candidate_id"] != failed["candidate_id"]
    }
    after_qm = record_quantum_results(
        path,
        _results(initial, accepted_ids),
        expected_generation=0,
    )

    replaced = allocate_replacements(
        path,
        replacement_round=1,
        expected_generation=int(after_qm["generation"]),
    )
    replacement = pending_attempts(replaced)

    assert len(replacement) == 1
    assert replacement[0]["slot_id"] == failed["slot_id"]
    assert replacement[0]["split"] == failed["split"]
    assert replacement[0]["round"] == 1


def test_quantum_result_batch_replay_is_idempotent_and_conflicts_fail(tmp_path):
    path, payload = _create(tmp_path)
    attempts = pending_attempts(payload)
    results = _results(attempts, {row["candidate_id"] for row in attempts})
    first = record_quantum_results(
        path,
        results,
        expected_generation=0,
        batch_identity="a" * 64,
        result_fingerprint="b" * 64,
    )
    replayed = record_quantum_results(
        path,
        results,
        batch_identity="a" * 64,
        result_fingerprint="b" * 64,
    )

    assert replayed == first
    assert replayed["generation"] == 1
    assert replayed["applied_quantum_batches"][0]["candidate_ids"] == sorted(
        row["candidate_id"] for row in attempts
    )
    with pytest.raises(ValueError, match="batch identity conflicts"):
        record_quantum_results(
            path,
            results,
            batch_identity="a" * 64,
            result_fingerprint="c" * 64,
        )


def test_parser_revalidation_preserves_rejected_generation_and_is_idempotent(
    tmp_path,
):
    path, payload = _create(
        tmp_path,
        targets={"train": 2, "int_val": 0, "ext_val": 0, "total": 2},
        n_reserve=0,
    )
    attempts = pending_attempts(payload)
    rejected = record_quantum_results(
        path,
        [
            {
                "candidate_id": attempt["candidate_id"],
                "accepted": False,
                "pointdir": "/staging/" + attempt["candidate_id"] + ".pointdir",
                "quality_manifest": None,
                "reason": "dft_model_missing_or_unsupported",
            }
            for attempt in attempts
        ],
    )
    corrections = [
        {
            "candidate_id": attempt["candidate_id"],
            "accepted": True,
            "prior_reason": "dft_model_missing_or_unsupported",
            "pointdir": attempt["pointdir"],
            "quality_manifest": "/staging/revalidated-quality.json",
            "quantum_acceptance_receipt": (
                "receipts/" + attempt["candidate_id"] + ".json"
            ),
            "quantum_acceptance_receipt_sha256": "a" * 64,
            "accepted_pointdir_content_sha256": "b" * 64,
        }
        for attempt in [slot["attempts"][-1] for slot in rejected["slots"]]
    ]

    with pytest.raises(ValueError, match="exactly the eligible vacant slots"):
        revalidate_rejected_quantum_results(
            path,
            corrections[:1],
            expected_generation=1,
            batch_identity="e" * 64,
            result_fingerprint="f" * 64,
        )

    updated = revalidate_rejected_quantum_results(
        path,
        corrections,
        expected_generation=1,
        batch_identity="c" * 64,
        result_fingerprint="d" * 64,
    )
    replayed = revalidate_rejected_quantum_results(
        path,
        corrections,
        expected_generation=1,
        batch_identity="c" * 64,
        result_fingerprint="d" * 64,
    )

    assert replayed == updated
    assert updated["generation"] == 2
    assert updated["summary"]["complete"] is True
    assert updated["applied_quantum_batches"][-1]["kind"] == "quality_revalidation"
    rejected_history = json.loads(
        (tmp_path / "history" / "generation-000001.json").read_text(
            encoding="utf-8"
        )
    )
    assert rejected_history["summary"]["deficit_total"] == 2
    assert all(
        slot["attempts"][-1]["status"] == "rejected"
        for slot in rejected_history["slots"]
    )


def test_parser_revalidation_preserves_148_existing_acceptance_bindings(tmp_path):
    path, payload = _create(
        tmp_path,
        targets={"train": 100, "int_val": 50, "ext_val": 0, "total": 150},
        n_reserve=0,
    )
    attempts = pending_attempts(payload)
    results = []
    for index, attempt in enumerate(attempts):
        accepted = index < 148
        result = {
            "candidate_id": attempt["candidate_id"],
            "accepted": accepted,
            "pointdir": "/staging/" + attempt["candidate_id"] + ".pointdir",
            "quality_manifest": "/staging/original-quality.json",
        }
        if accepted:
            result.update(
                {
                    "quantum_acceptance_receipt": (
                        "receipts/" + attempt["candidate_id"] + ".json"
                    ),
                    "quantum_acceptance_receipt_sha256": format(index + 1, "064x"),
                    "accepted_pointdir_content_sha256": format(index + 1000, "064x"),
                }
            )
        else:
            result["reason"] = "dft_model_missing_or_unsupported"
        results.append(result)
    rejected = record_quantum_results(path, results)
    original_acceptances = {
        attempt["candidate_id"]: dict(attempt)
        for attempt in accepted_attempts(rejected)
    }
    rejected_attempts = [
        slot["attempts"][-1]
        for slot in rejected["slots"]
        if slot.get("accepted_attempt") is None
    ]
    corrections = [
        {
            "candidate_id": attempt["candidate_id"],
            "accepted": True,
            "prior_reason": "dft_model_missing_or_unsupported",
            "pointdir": attempt["pointdir"],
            "quality_manifest": "/staging/revalidated-quality.json",
            "quantum_acceptance_receipt": (
                "receipts/" + attempt["candidate_id"] + "-revalidated.json"
            ),
            "quantum_acceptance_receipt_sha256": "a" * 64,
            "accepted_pointdir_content_sha256": "b" * 64,
        }
        for attempt in rejected_attempts
    ]

    updated = revalidate_rejected_quantum_results(
        path,
        corrections,
        expected_generation=1,
        batch_identity="c" * 64,
        result_fingerprint="d" * 64,
    )

    assert updated["summary"]["accepted_total"] == 150
    assert updated["summary"]["deficit_total"] == 0
    updated_by_id = {
        attempt["candidate_id"]: attempt for attempt in accepted_attempts(updated)
    }
    assert {
        candidate_id: updated_by_id[candidate_id]
        for candidate_id in original_acceptances
    } == original_acceptances


def test_replacement_completion_preserves_exact_counts(tmp_path):
    path, payload = _create(tmp_path)
    initial = pending_attempts(payload)
    after_qm = record_quantum_results(
        path,
        _results(initial, {initial[0]["candidate_id"]}),
    )
    replaced = allocate_replacements(
        path,
        replacement_round=1,
        expected_generation=int(after_qm["generation"]),
    )
    replacements = pending_attempts(replaced)
    complete = record_quantum_results(
        path,
        _results(replacements, {row["candidate_id"] for row in replacements}),
        expected_generation=int(replaced["generation"]),
    )

    assert complete["summary"]["complete"] is True
    assert complete["summary"]["accepted"] == {
        "train": 2,
        "int_val": 1,
        "ext_val": 0,
    }
    assert len(accepted_attempts(complete)) == 3


def test_mandatory_custom_failure_forbids_replacement(tmp_path):
    targets = {"train": 2, "int_val": 1, "ext_val": 1, "total": 4}
    path, payload = _create(
        tmp_path,
        context="bootstrap",
        targets=targets,
        mandatory=("candidate-0",),
        forced_splits={"candidate-0": "train"},
    )
    attempts = pending_attempts(payload)
    after_qm = record_quantum_results(
        path,
        _results(
            attempts,
            {
                attempt["candidate_id"]
                for attempt in attempts
                if attempt["candidate_id"] != "candidate-0"
            },
        ),
    )

    assert after_qm["mandatory_custom_failed"] is True
    with pytest.raises(ValueError, match="custom bootstrap geometry failed"):
        allocate_replacements(
            path,
            replacement_round=1,
            expected_generation=int(after_qm["generation"]),
        )


def test_mandatory_validation_geometry_failure_remains_readable(tmp_path):
    targets = {"train": 2, "int_val": 1, "ext_val": 1, "total": 4}
    path, payload = _create(
        tmp_path,
        context="bootstrap",
        targets=targets,
        mandatory=("candidate-0",),
        forced_splits={"candidate-0": "int_val"},
    )
    attempts = pending_attempts(payload)
    accepted = {
        attempt["candidate_id"]
        for attempt in attempts
        if attempt["candidate_id"] != "candidate-0"
    }

    after_qm = record_quantum_results(path, _results(attempts, accepted))

    assert after_qm["mandatory_custom_failed"] is True
    assert read_point_allocation(path)["mandatory_custom_failed"] is True
    with pytest.raises(ValueError, match="custom bootstrap geometry failed"):
        allocate_replacements(
            path,
            replacement_round=1,
            expected_generation=int(after_qm["generation"]),
        )


def test_reserve_exhaustion_fails_without_partial_allocation(tmp_path):
    path, payload = _create(tmp_path, n_reserve=1)
    attempts = pending_attempts(payload)
    after_qm = record_quantum_results(path, _results(attempts, set()))

    with pytest.raises(ValueError, match="reserve exhausted"):
        allocate_replacements(
            path,
            replacement_round=1,
            expected_generation=int(after_qm["generation"]),
        )
    unchanged = read_point_allocation(path)
    assert unchanged["generation"] == after_qm["generation"]
    assert unchanged["summary"]["reserve_consumed"] == 0


def test_quantum_result_retry_is_idempotent(tmp_path):
    path, payload = _create(tmp_path)
    attempts = pending_attempts(payload)
    results = _results(attempts, {row["candidate_id"] for row in attempts})
    first = record_quantum_results(path, results)
    second = record_quantum_results(
        path,
        results,
        expected_generation=int(first["generation"]),
    )

    assert second == first
    assert second["generation"] == 1
    history = tmp_path / "history" / "generation-000000.json"
    assert history.is_file()


def test_history_chain_is_verified_on_read(tmp_path):
    path, payload = _create(tmp_path)
    attempts = pending_attempts(payload)
    after_qm = record_quantum_results(path, _results(attempts, set()))
    allocate_replacements(
        path,
        replacement_round=1,
        expected_generation=int(after_qm["generation"]),
    )

    assert read_point_allocation(path)["generation"] == 2

    history = tmp_path / "history" / "generation-000000.json"
    corrupted = json.loads(history.read_text(encoding="utf-8"))
    corrupted["campaign_uid"] = "tampered-campaign"
    history.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="predecessor"):
        read_point_allocation(path)


def test_crash_window_archive_of_current_generation_is_retryable(tmp_path):
    path, payload = _create(tmp_path)
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "generation-000000.json").write_text(
        path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert read_point_allocation(path) == payload
    attempts = pending_attempts(payload)
    updated = record_quantum_results(
        path,
        _results(attempts, {row["candidate_id"] for row in attempts}),
    )
    assert updated["generation"] == 1


def test_create_retry_accepts_same_candidate_universe_after_mutation(tmp_path):
    path, payload = _create(tmp_path)
    attempts = pending_attempts(payload)
    record_quantum_results(
        path,
        _results(attempts, {row["candidate_id"] for row in attempts}),
    )

    existing = create_point_allocation(
        path,
        campaign_uid="campaign-uid",
        context="active",
        iteration=1,
        targets={"train": 2, "int_val": 1, "ext_val": 0, "total": 3},
        primary_candidates=[_candidate(i) for i in range(3)],
        reserve_candidates=[_candidate(3 + i, reserve=True) for i in range(3)],
    )

    assert existing["generation"] == 1
    assert existing["summary"]["complete"] is True


def test_create_retry_rejects_changed_candidate_evidence(tmp_path):
    path, _payload = _create(tmp_path)
    changed_primary = [_candidate(i) for i in range(3)]
    changed_primary[1]["provenance_sha256"] = "changed-after-first-allocation"

    with pytest.raises(ValueError, match="refusing to replace"):
        create_point_allocation(
            path,
            campaign_uid="campaign-uid",
            context="active",
            iteration=1,
            targets={"train": 2, "int_val": 1, "ext_val": 0, "total": 3},
            primary_candidates=changed_primary,
            reserve_candidates=[
                _candidate(3 + i, reserve=True) for i in range(3)
            ],
        )


def test_candidate_diagnostics_reject_non_finite_numbers(tmp_path):
    path = tmp_path / "POINT_ALLOCATION.json"

    with pytest.raises(ValueError, match="non-finite"):
        create_point_allocation(
            path,
            campaign_uid="campaign-uid",
            context="active",
            iteration=1,
            targets={"train": 1, "int_val": 0, "ext_val": 0, "total": 1},
            primary_candidates=[
                {
                    "candidate_id": "candidate-0",
                    "distance_to_nearest_angstrom": float("inf"),
                    "nested": {"score": float("nan")},
                }
            ],
            reserve_candidates=[],
        )


def test_consumed_reserve_round_must_match_exact_replacement_attempt(tmp_path):
    path, payload = _create(tmp_path)
    initial = pending_attempts(payload)
    rejected = record_quantum_results(path, _results(initial, set()))
    allocate_replacements(
        path,
        replacement_round=1,
        expected_generation=int(rejected["generation"]),
    )
    tampered = json.loads(path.read_text(encoding="utf-8"))
    consumed = next(
        record for record in tampered["reserve"] if record["status"] == "consumed"
    )
    consumed["consumed_round"] = 2
    path.write_text(json.dumps(tampered), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="consumed reserve round"):
        read_point_allocation(path)


def test_allocation_reader_rejects_stale_derived_summary(tmp_path):
    path, _payload = _create(tmp_path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["summary"]["pending"] = 0
    path.write_text(json.dumps(tampered), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="summary is inconsistent"):
        read_point_allocation(path)


@pytest.mark.parametrize("bad", [1.5, "1", True, None])
def test_allocation_reader_rejects_non_integer_generation(tmp_path, bad):
    path, _payload = _create(tmp_path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["generation"] = bad
    path.write_text(json.dumps(tampered), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="exact JSON integer"):
        read_point_allocation(path)


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_quantum_results_require_exact_boolean_acceptance(tmp_path, bad):
    path, payload = _create(tmp_path)
    result = _results(pending_attempts(payload), set())
    result[0]["accepted"] = bad
    with pytest.raises(ValueError, match="exact JSON Boolean"):
        record_quantum_results(path, result)


def test_accepted_attempt_requires_pointdir_and_quality_evidence(tmp_path):
    path, payload = _create(tmp_path)
    results = _results(
        pending_attempts(payload),
        {attempt["candidate_id"] for attempt in pending_attempts(payload)},
    )
    results[0].pop("quality_manifest")
    with pytest.raises(ValueError, match="quality-manifest evidence"):
        record_quantum_results(path, results)


def test_reader_rejects_accepted_attempt_without_pointdir(tmp_path):
    path, payload = _create(tmp_path)
    results = _results(
        pending_attempts(payload),
        {attempt["candidate_id"] for attempt in pending_attempts(payload)},
    )
    record_quantum_results(path, results)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    accepted = next(
        attempt
        for slot in tampered["slots"]
        for attempt in slot["attempts"]
        if attempt["status"] == "accepted"
    )
    accepted.pop("pointdir")
    tampered.pop("summary", None)
    path.write_text(json.dumps(tampered), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="invalid pointdir"):
        read_point_allocation(path)
