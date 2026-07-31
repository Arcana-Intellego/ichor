"""Scientific environment-equivalence proofs for cancelled task reuse."""

import json
from pathlib import Path
import subprocess

from ichor.hpc.active_learning.daemon import environment_equivalence as module
from ichor.hpc.active_learning import execution_identity
from ichor.hpc.active_learning.daemon.state import atomic_write_json


def _intent():
    return {
        "campaign_uid": "campaign-environment-equivalence",
        "phase": "ARIADNE_ARRAY",
        "iteration": 4,
        "replacement_round": 0,
        "attempt_id": "attempt0001",
        "submission_identity": "r0000-a0001-attempt1",
        "scheduler_identity_kind": "slurm",
        "job_id": "12345",
        "environment_generation": 2,
        "environment_generation_digest_sha256": "a" * 64,
        "resource_resolution_path": ".DATA/ACTIVE_LEARNING/resources.json",
        "resource_resolution_sha256": "c" * 64,
    }


def _generation(
    *,
    generation,
    digest,
    dependencies=None,
    loaded_modules=None,
    native_library_paths=None,
    module_sequence=None,
):
    return {
        "generation": generation,
        "digest_sha256": digest,
        "python_executable": "/opt/python/bin/python",
        "python_version": "3.11.15",
        "dependencies": (
            {"numpy": "1.26.4"}
            if dependencies is None
            else dependencies
        ),
        "loaded_modules": (
            ["python/runtime"]
            if loaded_modules is None
            else loaded_modules
        ),
        "native_library_paths": (
            {
                "LD_LIBRARY_PATH": "/opt/runtime/lib",
                "LIBRARY_PATH": "/opt/runtime/lib",
            }
            if native_library_paths is None
            else native_library_paths
        ),
        "machine_profile": {
            "module_sequence": (
                {
                    "purge_first": True,
                    "python_modules": ["python/runtime"],
                    "ariadne_runtime_modules": ["intel/runtime"],
                }
                if module_sequence is None
                else module_sequence
            )
        },
        "ichor_git": {
            "commit": "1" * 40,
            "tracked_tree_clean": True,
        },
    }


def _patch_environment_readers(
    monkeypatch,
    *,
    producer,
    current,
):
    monkeypatch.setattr(
        execution_identity,
        "read_environment_generation",
        lambda *_args, **_kwargs: dict(producer),
    )
    monkeypatch.setattr(
        execution_identity,
        "read_active_environment_generation",
        lambda *_args, **_kwargs: {"generation": dict(current)},
    )
    monkeypatch.setattr(
        module,
        "_resolution_for_intent",
        lambda *_args, **_kwargs: {"implementation_identity": {}},
    )
    monkeypatch.setattr(
        module,
        "_native_identity_reasons",
        lambda *_args, **_kwargs: [],
    )


def test_identical_scientific_environment_is_reusable(tmp_path, monkeypatch):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(generation=3, digest="a" * 64)
    _patch_environment_readers(
        monkeypatch,
        producer=producer,
        current=current,
    )

    assessment = module.assess_recovery_environment(
        tmp_path,
        intent=_intent(),
        current_scheduler_kind="slurm",
    )

    assert assessment["equivalent"] is True
    assert assessment["reasons"] == []
    assert Path(assessment["path"]).is_file()


def test_legacy_equivalence_proof_remains_readable_but_is_separate(tmp_path, monkeypatch):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(generation=3, digest="a" * 64)
    _patch_environment_readers(
        monkeypatch,
        producer=producer,
        current=current,
    )
    assessment = module.assess_recovery_environment(
        tmp_path,
        intent=_intent(),
        current_scheduler_kind="slurm",
    )
    current_path = Path(assessment["path"])
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    payload["schema_version"] = (
        module.LEGACY_ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION
    )
    payload.pop("fingerprint_algorithm", None)
    payload.pop("uncovered_runtime_changes", None)
    payload.pop("unresolved_dynamic_import_modules", None)
    payload.pop("proof_sha256", None)
    payload["proof_sha256"] = module._sha256_json(payload)
    legacy_path = module.environment_equivalence_path(
        tmp_path,
        phase=payload["phase"],
        iteration=payload["iteration"],
        replacement_round=payload["replacement_round"],
        submission_identity=payload["submission_identity"],
        current_generation=payload["current_generation"],
        schema_version=module.LEGACY_ENVIRONMENT_EQUIVALENCE_SCHEMA_VERSION,
    )
    atomic_write_json(legacy_path, payload)

    assert module.read_environment_equivalence(legacy_path) == payload
    assert current_path != legacy_path


def test_changed_producer_code_forces_retry(tmp_path, monkeypatch):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(generation=3, digest="b" * 64)
    _patch_environment_readers(
        monkeypatch,
        producer=producer,
        current=current,
    )
    monkeypatch.setattr(module, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_require_clean_available_commit",
        lambda *_args, label, **_kwargs: label + "-commit",
    )
    monkeypatch.setattr(
        module,
        "_producer_fingerprint",
        lambda _repo, commit, backend: {
            "algorithm": module.SCIENTIFIC_FINGERPRINT_ALGORITHM,
            "backend": backend,
            "modules": [],
            "covered_paths": [],
            "fingerprint_sha256": (
                "1" * 64 if commit == "producer-commit" else "2" * 64
            ),
        },
    )
    monkeypatch.setattr(
        module,
        "_uncovered_runtime_changes",
        lambda *_args, **_kwargs: (),
    )

    assessment = module.assess_recovery_environment(
        tmp_path,
        intent=_intent(),
        current_scheduler_kind="slurm",
    )

    assert assessment["equivalent"] is False
    assert "scientific_producer_code_changed" in assessment["reasons"]


def test_dependency_or_scheduler_change_forces_retry(tmp_path, monkeypatch):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(
        generation=3,
        digest="a" * 64,
        dependencies={"numpy": "2.0.0"},
    )
    _patch_environment_readers(
        monkeypatch,
        producer=producer,
        current=current,
    )

    assessment = module.assess_recovery_environment(
        tmp_path,
        intent=_intent(),
        current_scheduler_kind="sge",
    )

    assert assessment["equivalent"] is False
    assert "scheduler_kind_changed" in assessment["reasons"]
    assert "dependency_environment_changed" in assessment["reasons"]


def test_native_runtime_change_forces_retry(tmp_path, monkeypatch):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(
        generation=3,
        digest="a" * 64,
        loaded_modules=["different/runtime"],
        native_library_paths={
            "LD_LIBRARY_PATH": "/different/lib",
            "LIBRARY_PATH": "/different/lib",
        },
        module_sequence={
            "purge_first": True,
            "python_modules": ["different/runtime"],
            "ariadne_runtime_modules": ["different/intel"],
        },
    )
    _patch_environment_readers(
        monkeypatch,
        producer=producer,
        current=current,
    )

    assessment = module.assess_recovery_environment(
        tmp_path,
        intent=_intent(),
        current_scheduler_kind="slurm",
    )

    assert assessment["equivalent"] is False
    assert assessment["reasons"] == [
        "loaded_modules_changed",
        "native_library_paths_changed",
        "profile_module_sequence_changed",
    ]


def test_resource_evidence_equivalence_uses_scientific_algorithm_roots(
    tmp_path,
    monkeypatch,
):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(generation=3, digest="b" * 64)
    monkeypatch.setattr(module, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_require_clean_available_commit",
        lambda *_args, label, **_kwargs: label + "-commit",
    )
    calls = []

    def fingerprint(_repo, commit, backend, roots):
        calls.append((commit, backend, roots))
        return {
            "algorithm": module.SCIENTIFIC_FINGERPRINT_ALGORITHM,
            "backend": backend,
            "modules": [],
            "covered_paths": [],
            "fingerprint_sha256": "1" * 64,
        }

    monkeypatch.setattr(module, "_fingerprint_roots", fingerprint)
    monkeypatch.setattr(
        module,
        "_uncovered_runtime_changes",
        lambda *_args, **_kwargs: (),
    )

    assessment = module.assess_resource_evidence_code_equivalence(
        producer,
        current,
        backend="ariadne",
    )

    assert assessment["equivalent"] is True
    assert [call[0] for call in calls] == [
        "producer-commit",
        "current-commit",
    ]
    flattened = {
        (path, symbol)
        for _commit, _backend, roots in calls
        for path, symbols in roots
        for symbol in symbols
    }
    assert (
        "ichor_core/ichor/core/adversarial/geometry.py",
        "aligned_mass_weighted_distance",
    ) in flattened
    assert (
        "ichor_core/ichor/core/adversarial/subspace.py",
        "build_local_subspace",
    ) in flattened


def test_unresolved_dynamic_repository_import_blocks_equivalence(
    tmp_path,
    monkeypatch,
):
    producer = _generation(generation=2, digest="a" * 64)
    current = _generation(generation=3, digest="b" * 64)
    monkeypatch.setattr(module, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        module,
        "_require_clean_available_commit",
        lambda *_args, label, **_kwargs: label + "-commit",
    )
    monkeypatch.setattr(
        module,
        "_fingerprint_roots",
        lambda _repo, _commit, backend, _roots: {
            "algorithm": module.SCIENTIFIC_FINGERPRINT_ALGORITHM,
            "backend": backend,
            "modules": [],
            "covered_paths": [],
            "unresolved_dynamic_import_modules": [
                "ichor_hpc/ichor/hpc/active_learning/producer.py"
            ],
            "fingerprint_sha256": "1" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "_uncovered_runtime_changes",
        lambda *_args, **_kwargs: (),
    )

    assessment = module.assess_resource_evidence_code_equivalence(
        producer,
        current,
        backend="ariadne",
    )

    assert assessment["equivalent"] is False
    assert assessment["unresolved_dynamic_import_modules"]


def test_v2_producer_closure_covers_audit_escape_dependencies():
    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    ariadne = module._producer_fingerprint(repo, commit, "ariadne")
    gaussian = module._producer_fingerprint(repo, commit, "gaussian")

    assert ariadne["algorithm"] == module.SCIENTIFIC_FINGERPRINT_ALGORITHM
    assert "ichor_core/ichor/core/atoms/atoms.py" in ariadne["covered_paths"]
    assert (
        "ichor_core/ichor/core/adversarial/geometry.py"
        in ariadne["covered_paths"]
    )
    assert (
        "ichor_core/ichor/core/files/gaussian/gjf.py"
        in gaussian["covered_paths"]
    )


def test_uncovered_runtime_change_blocks_equivalence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "_changed_paths",
        lambda *_args, **_kwargs: {
            "ichor_core/ichor/core/unresolved_dynamic_dependency.py",
            "docs/recovery.md",
        },
    )

    uncovered = module._uncovered_runtime_changes(
        tmp_path,
        "producer",
        "current",
        {"covered_paths": ["ichor_hpc/ichor/hpc/active_learning/producer.py"]},
    )

    assert uncovered == (
        "ichor_core/ichor/core/unresolved_dynamic_dependency.py",
    )
