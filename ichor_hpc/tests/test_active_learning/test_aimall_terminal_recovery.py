import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ichor.hpc.active_learning.daemon.aimall_output_validation import (
    AIMALL_OUTPUT_INVALID,
    AIMAllOutputAssessment,
)
from ichor.hpc.active_learning.daemon.aimall_terminal_recovery import (
    AIMALL_STRUCTURAL_RETRY_METADATA_KEY,
    AIMALL_STRUCTURAL_VALIDATOR_METADATA_KEY,
    aimall_intent_has_structural_validator_contract,
    build_aimall_structural_validator_metadata,
    build_aimall_structural_retry_metadata,
    classify_aimall_postprocess_source_outputs,
    resolve_aimall_structural_recovery_policy,
    validate_aimall_structural_retry_metadata,
)


UID = "fixture-campaign"
PHASE = "AIMALL"


def _source(identity="r0000-a0001-source", job_id="898504"):
    return {
        "kind": "scheduler_terminal_receipt",
        "sha256": "a" * 64,
        "submission_identity": identity,
        "job_id": job_id,
    }


def _intent(sequence, identity, *, metadata=None, job_id="898504"):
    payload = {
        "campaign_uid": UID,
        "phase": PHASE,
        "iteration": 8,
        "replacement_round": 0,
        "attempt_sequence": sequence,
        "submission_identity": identity,
        "job_id": job_id,
    }
    if metadata is not None:
        payload["submission_metadata"] = {
            AIMALL_STRUCTURAL_RETRY_METADATA_KEY: metadata,
        }
    return payload


def _receipt(identity, outcomes, job_id="898504"):
    return {
        "receipt_sha256": ("b" if identity.endswith("1") else "c") * 64,
        "submission_identity": identity,
        "job_id": job_id,
        "outcomes": outcomes,
    }


def _outcome(task_id, status, exit_code):
    return {
        "logical_task_id": task_id,
        "status": status,
        "exit_code": exit_code,
    }


def _install_contract(monkeypatch, tmp_path, intents):
    pointdirs = []
    for task_id in range(2):
        pointdir = tmp_path / ("POINT_" + str(task_id).zfill(4) + ".pointdir")
        pointdir.mkdir(parents=True)
        pointdirs.append(pointdir)
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.submission_intent."
        "intent_attempt_records",
        lambda *_args, **_kwargs: list(intents),
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.quantum_task_contracts."
        "quantum_task_contract",
        lambda *_args, **_kwargs: SimpleNamespace(
            logical_total=2,
            tasks=[
                SimpleNamespace(logical_task_id=index, pointdir=path)
                for index, path in enumerate(pointdirs)
            ],
        ),
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.aimall_terminal_recovery."
        "aimall_intent_has_structural_validator_contract",
        lambda *_args, **_kwargs: True,
    )
    return pointdirs


def test_validator_contract_is_bound_to_submitted_script(tmp_path):
    script = tmp_path / "aimall.sh"
    script.write_text(
        "#!/bin/bash\n"
        'if [[ "$ICHOR_AIMALL_BACKEND_STATUS" -eq 86 ]]; then exit 85; fi\n'
        "python -m ichor.hpc.active_learning.daemon."
        "aimall_output_validation --pointdir x\n",
        encoding="utf-8",
    )
    script_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
    binding = tmp_path / "script-binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "script_path": str(script),
                "script_size": script.stat().st_size,
                "script_sha256": script_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    binding_sha256 = hashlib.sha256(binding.read_bytes()).hexdigest()
    intent = {
        "submitted_script_path": str(script),
        "submitted_script_sha256": script_sha256,
        "script_binding_path": str(binding),
        "script_binding_sha256": binding_sha256,
        "submission_metadata": {
            AIMALL_STRUCTURAL_VALIDATOR_METADATA_KEY: (
                build_aimall_structural_validator_metadata(script_sha256)
            )
        },
    }

    assert aimall_intent_has_structural_validator_contract(
        tmp_path,
        intent,
    ) is True

    script.write_text("#!/bin/bash\nexit 86\n", encoding="utf-8")
    with pytest.raises(ValueError, match="script.*drifted"):
        aimall_intent_has_structural_validator_contract(tmp_path, intent)


def test_structural_retry_metadata_is_self_authenticating():
    metadata = build_aimall_structural_retry_metadata(
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        task_ids=[1],
        source_records=[_source()],
    )

    assert validate_aimall_structural_retry_metadata(
        metadata,
        expected_campaign_uid=UID,
        expected_phase=PHASE,
        expected_iteration=8,
        expected_replacement_round=0,
    ) == metadata

    changed = dict(metadata)
    changed["task_ids"] = [0]
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_aimall_structural_retry_metadata(
            changed,
            expected_campaign_uid=UID,
            expected_phase=PHASE,
            expected_iteration=8,
            expected_replacement_round=0,
        )


def test_first_exit_86_retries_once_and_second_becomes_rejection(
    tmp_path,
    monkeypatch,
):
    first_identity = "r0000-a0001"
    first_intent = _intent(1, first_identity)
    first_receipt = _receipt(
        first_identity,
        [
            _outcome(0, "COMPLETED", [0, 0]),
            _outcome(1, "FAILED", [86, 0]),
        ],
    )
    _install_contract(monkeypatch, tmp_path, [first_intent])

    first = resolve_aimall_structural_recovery_policy(
        tmp_path,
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        recoveries=[{"intent": first_intent, "receipt": first_receipt}],
        invalid_completed_tasks=[
            {
                "task_id": 1,
                "reason": "int_file_incomplete:h3.int",
                "classification": "structural",
            }
        ],
    )

    assert first["structural_retry_task_ids"] == [1]
    assert first["terminal_rejection_task_ids"] == []

    retry_metadata = build_aimall_structural_retry_metadata(
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        task_ids=[1],
        source_records=first["source_records"],
    )
    retry_identity = "r0000-a0002"
    retry_intent = _intent(
        2,
        retry_identity,
        metadata=retry_metadata,
        job_id="898505",
    )
    retry_receipt = _receipt(
        retry_identity,
        [_outcome(1, "FAILED", [86, 0])],
        job_id="898505",
    )
    pointdirs = _install_contract(
        monkeypatch,
        tmp_path / "replay",
        [first_intent, retry_intent],
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.aimall_terminal_recovery."
        "assess_aimall_output",
        lambda path: AIMAllOutputAssessment(
            AIMALL_OUTPUT_INVALID,
            "int_file_incomplete:h3.int",
            "d" * 64,
        ),
    )

    second = resolve_aimall_structural_recovery_policy(
        tmp_path / "replay",
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        recoveries=[
            {"intent": first_intent, "receipt": first_receipt},
            {"intent": retry_intent, "receipt": retry_receipt},
        ],
        invalid_completed_tasks=[
            {
                "task_id": 1,
                "reason": "int_file_incomplete:h3.int",
                "classification": "structural",
            }
        ],
    )

    assert pointdirs[1].is_dir()
    assert second["structural_retry_task_ids"] == []
    assert second["terminal_rejection_task_ids"] == [1]
    assert second["terminal_rejection_reasons"] == {
        "1": "int_file_incomplete:h3.int"
    }


def test_unmarked_legacy_exit_86_is_not_a_structural_failure(
    tmp_path,
    monkeypatch,
):
    identity = "r0000-a0001-legacy"
    intent = _intent(1, identity)
    receipt = _receipt(
        identity,
        [_outcome(1, "FAILED", [86, 0])],
    )
    _install_contract(monkeypatch, tmp_path, [intent])
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.aimall_terminal_recovery."
        "aimall_intent_has_structural_validator_contract",
        lambda *_args, **_kwargs: False,
    )

    policy = resolve_aimall_structural_recovery_policy(
        tmp_path,
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        recoveries=[{"intent": intent, "receipt": receipt}],
    )

    assert policy["structural_retry_task_ids"] == []
    assert policy["terminal_rejection_task_ids"] == []


def test_infrastructure_failure_does_not_consume_structural_retry(
    tmp_path,
    monkeypatch,
):
    metadata = build_aimall_structural_retry_metadata(
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        task_ids=[1],
        source_records=[_source()],
    )
    intent = _intent(
        2,
        "r0000-a0002",
        metadata=metadata,
        job_id="898505",
    )
    receipt = _receipt(
        "r0000-a0002",
        [_outcome(1, "FAILED", [1, 0])],
        job_id="898505",
    )
    _install_contract(monkeypatch, tmp_path, [intent])

    policy = resolve_aimall_structural_recovery_policy(
        tmp_path,
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        recoveries=[{"intent": intent, "receipt": receipt}],
    )

    assert policy["structural_retry_task_ids"] == [1]
    assert policy["terminal_rejection_task_ids"] == []


def test_iteration_eight_classification_binds_149_original_receipts(
    tmp_path,
    monkeypatch,
):
    pointdirs = []
    tasks = []
    for task_id in range(150):
        pointdir = tmp_path / (
            "POINT_" + str(task_id).zfill(4) + ".pointdir"
        )
        pointdir.mkdir()
        pointdirs.append(pointdir)
        tasks.append(
            SimpleNamespace(
                logical_task_id=task_id,
                pointdir=pointdir,
            )
        )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.quantum_task_contracts."
        "quantum_task_contract",
        lambda *_args, **_kwargs: SimpleNamespace(
            logical_total=150,
            tasks=tasks,
        ),
    )

    def assess(path):
        task_id = pointdirs.index(Path(path))
        if task_id == 88:
            return AIMAllOutputAssessment(
                AIMALL_OUTPUT_INVALID,
                "int_file_incomplete:h3.int",
                "8" * 64,
            )
        return AIMAllOutputAssessment("valid", "", format(task_id, "064x"))

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.aimall_terminal_recovery."
        "assess_aimall_output",
        assess,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.input_staging."
        "validate_existing_aimall_task_authorities",
        lambda *_args, **_kwargs: None,
    )
    published = []
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.quantum_task_receipts."
        "write_quantum_task_receipt_from_postprocess_source",
        lambda pointdir, **kwargs: published.append(
            (Path(pointdir).name, dict(kwargs))
        ),
    )
    source = {
        "logical_total": 150,
        "job_id": "898504",
        "submission_identity": "r0000-a0001-original",
        "source_sha256": "f" * 64,
    }

    classification = classify_aimall_postprocess_source_outputs(
        tmp_path,
        campaign_uid=UID,
        phase_name=PHASE,
        iteration=8,
        replacement_round=0,
        source=source,
        expected_method="B3LYP",
        publish_valid_receipts=True,
    )

    assert classification["scheduler_completed_candidates"] == 150
    assert classification["validated_reusable_task_ids"] == [
        task_id for task_id in range(150) if task_id != 88
    ]
    assert classification["invalid_completed_tasks"] == [
        {
            "task_id": 88,
            "reason": "int_file_incomplete:h3.int",
            "fingerprint_sha256": "8" * 64,
        }
    ]
    assert classification["producer_job_id"] == "898504"
    assert len(published) == 149
    assert {record[1]["logical_task_id"] for record in published} == (
        set(range(150)) - {88}
    )
    assert all(record[1]["source"] is source for record in published)


def test_postprocess_classification_blocks_inherited_authority_drift(
    tmp_path,
    monkeypatch,
):
    _install_contract(monkeypatch, tmp_path, [])

    def reject_drift(*_args, **_kwargs):
        raise ValueError("WFN receipt digest mismatch")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.input_staging."
        "validate_existing_aimall_task_authorities",
        reject_drift,
    )
    published = []
    monkeypatch.setattr(
        "ichor.hpc.active_learning.daemon.quantum_task_receipts."
        "write_quantum_task_receipt_from_postprocess_source",
        lambda *_args, **_kwargs: published.append(dict(_kwargs)),
    )

    with pytest.raises(ValueError, match="WFN receipt digest mismatch"):
        classify_aimall_postprocess_source_outputs(
            tmp_path,
            campaign_uid=UID,
            phase_name=PHASE,
            iteration=8,
            replacement_round=0,
            source={
                "logical_total": 2,
                "job_id": "898504",
                "submission_identity": "r0000-a0001-original",
                "source_sha256": "f" * 64,
            },
            expected_method="B3LYP",
            publish_valid_receipts=True,
        )

    assert published == []
