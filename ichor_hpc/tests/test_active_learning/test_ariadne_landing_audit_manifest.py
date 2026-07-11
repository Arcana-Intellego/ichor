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
    iter_dir = tmp_path / "iteration-000001"
    payload = {
        "iteration": 1,
        "summary": {
            "accepted": 1,
            "salvaged": 0,
            "backtracked": 0,
            "rejected": 0,
            "rejection_reasons": {},
            "policies": {"raw_final": 1},
        },
        "seeds": [{
            "seed_id": 1,
            "seed_uid": "seed-uid-1",
            "seed_dir": "seeds/seed-000001",
            "landing_safety": {
                "accepted": True,
                "policy": "raw_final",
                "reasons": [],
                "metrics": {"whitened_distance": 0.5},
            },
        }],
    }
    path = write_ariadne_landing_audit(iter_dir, payload)
    assert path.name == "AUDIT.json"
    loaded = read_ariadne_landing_audit(iter_dir, expected_iteration=1)
    assert loaded["schema_version"] == 2
    assert loaded["summary"]["accepted"] == 1
    assert loaded["seeds"][0]["seed_id"] == 1


def test_ariadne_landing_audit_rejects_duplicate_seed_id(tmp_path):
    iter_dir = tmp_path / "iteration-000001"
    write_ariadne_landing_audit(iter_dir, {
        "iteration": 1,
        "summary": {},
        "seeds": [{"seed_id": 1}, {"seed_id": 1}],
    })
    with pytest.raises(HandoffManifestError, match="duplicate"):
        read_ariadne_landing_audit(iter_dir, expected_iteration=1)


def test_acquisition_maturity_audit_roundtrip(tmp_path):
    iter_dir = tmp_path / "iteration-000001"
    payload = acquisition_maturity_audit_payload(
        iteration=1,
        seed_records=[{
            "seed_id": 1,
            "seed_uid": "seed-uid-1",
            "result_json": "seeds/seed-000001/result.json",
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
    assert path.name == "AUDIT.json"
    loaded = read_acquisition_maturity_audit(iter_dir, expected_iteration=1)
    assert loaded["schema_version"] == 2
    maturity = loaded["maturity"]
    assert maturity["summary"]["n_candidates_with_spectral"] == 1
    assert maturity["summary"]["n_candidates_with_fullspace_residual"] == 1
    assert maturity["summary"]["n_candidates_with_banded_energy"] == 1
    assert maturity["seeds"][0]["landing_candidates"][0]["metrics"]["total_score"] == 1.0
