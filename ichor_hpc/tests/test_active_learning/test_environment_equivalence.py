"""Scientific environment-equivalence proofs for cancelled task reuse."""

from pathlib import Path

from ichor.hpc.active_learning.daemon import environment_equivalence as module
from ichor.hpc.active_learning import execution_identity


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
            "backend": backend,
            "symbols": [],
            "fingerprint_sha256": (
                "1" * 64 if commit == "producer-commit" else "2" * 64
            ),
        },
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
            "backend": backend,
            "symbols": [],
            "fingerprint_sha256": "1" * 64,
        }

    monkeypatch.setattr(module, "_fingerprint_roots", fingerprint)

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
