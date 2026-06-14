import pytest

from ichor.hpc.active_learning.handoff_manifests import (
    HandoffManifestError,
    acquisition_maturity_audit_payload,
    read_acquisition_maturity_audit,
    read_ariadne_landing_audit,
    write_acquisition_maturity_audit,
    write_ariadne_landing_audit,
)


def test_ariadne_landing_audit_roundtrip(tmp_path):
    iter_dir = tmp_path / "iteration-0000"
    payload = {
        "iteration": 0,
        "summary": {
            "accepted": 1,
            "salvaged": 0,
            "backtracked": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "policies": {"raw_final": 1},
        },
        "seeds": [{
            "seed_index": 0,
            "seed_dir": str((iter_dir / "pool" / "seed_0000").resolve()),
            "landing_safety": {
                "accepted": True,
                "policy": "raw_final",
                "reasons": [],
                "metrics": {"whitened_distance": 0.5},
            },
        }],
    }
    path = write_ariadne_landing_audit(iter_dir, payload)
    assert path.name == "ARIADNE_LANDING_AUDIT.json"
    loaded = read_ariadne_landing_audit(iter_dir, expected_iteration=0)
    assert loaded["schema_version"] == 1
    assert loaded["summary"]["accepted"] == 1
    assert loaded["seeds"][0]["seed_index"] == 0


def test_ariadne_landing_audit_rejects_duplicate_seed_index(tmp_path):
    iter_dir = tmp_path / "iteration-0000"
    write_ariadne_landing_audit(iter_dir, {
        "iteration": 0,
        "summary": {},
        "seeds": [{"seed_index": 0}, {"seed_index": 0}],
    })
    with pytest.raises(HandoffManifestError, match="duplicate"):
        read_ariadne_landing_audit(iter_dir, expected_iteration=0)


def test_acquisition_maturity_audit_roundtrip(tmp_path):
    iter_dir = tmp_path / "iteration-0000"
    payload = acquisition_maturity_audit_payload(
        iteration=0,
        seed_records=[{
            "seed_index": 0,
            "result_json": str((iter_dir / "pool" / "seed_0000" / "result.json").resolve()),
            "landing_safety": {
                "policy": "raw_final",
                "metrics": {"whitened_distance": 0.5},
            },
            "selection_diagnostics": {
                "spectral_frequency_risk": 0.7,
                "fullspace_residual_distance": 0.02,
            },
            "landing_candidates": [{
                "candidate_index": 0,
                "origin": "raw_final",
                "accepted": True,
                "metrics": {
                    "total_score": 1.0,
                    "spectral_frequency_risk": 0.7,
                    "fullspace_residual_distance": 0.02,
                    "banded_energy_risk": 0.3,
                    "acquisition_fallback_reasons": ["banded_energy_inferred_from_calibration_model"],
                },
            }],
        }],
    )
    path = write_acquisition_maturity_audit(iter_dir, payload)
    assert path.name == "ACQUISITION_MATURITY_AUDIT.json"
    loaded = read_acquisition_maturity_audit(iter_dir, expected_iteration=0)
    assert loaded["schema_version"] == 1
    assert loaded["summary"]["n_candidates_with_spectral"] == 1
    assert loaded["summary"]["n_candidates_with_fullspace_residual"] == 1
    assert loaded["summary"]["n_candidates_with_banded_energy"] == 1
    assert loaded["seeds"][0]["landing_candidates"][0]["metrics"]["total_score"] == 1.0
