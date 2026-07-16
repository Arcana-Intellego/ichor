"""ARIADNE geometry sanity-gate tests."""
from types import SimpleNamespace

from ichor.hpc.active_learning.daemon.live_executor import _ariadne_geometry_quality


def test_ariadne_geometry_quality_records_finite_metrics():
    payload = {
        "initial_coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    validated = {
        "final_coordinates": [[0.1, 0.0, 0.0], [1.0, 0.2, 0.0]],
    }

    result = _ariadne_geometry_quality(
        payload,
        validated,
        SimpleNamespace(
            ariadne_max_displacement_angstrom=1.25,
            ariadne_min_pair_distance_angstrom=0.60,
        ),
    )

    assert result["accepted"] is True
    assert result["reasons"] == []
    assert result["metrics"]["max_displacement_ang"] > 0.19
    assert result["metrics"]["min_pair_distance_ang"] > 0.9


def test_ariadne_geometry_quality_rejects_nonfinite_coordinates():
    result = _ariadne_geometry_quality(
        {"initial_coordinates": [[0.0, 0.0, 0.0]]},
        {"final_coordinates": [[float("nan"), 0.0, 0.0]]},
        SimpleNamespace(),
    )

    assert result["accepted"] is False
    assert "ariadne_final_geometry_nonfinite" in result["reasons"]


def test_ariadne_geometry_quality_enforces_optional_thresholds():
    payload = {
        "initial_coordinates": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }
    validated = {
        "final_coordinates": [[0.5, 0.0, 0.0], [0.55, 0.0, 0.0]],
    }

    result = _ariadne_geometry_quality(
        payload,
        validated,
        SimpleNamespace(
            ariadne_max_displacement_angstrom=0.1,
            ariadne_min_pair_distance_angstrom=0.2,
        ),
    )

    assert result["accepted"] is False
    assert "ariadne_max_displacement_threshold_exceeded" in result["reasons"]
    assert "ariadne_min_pair_distance_threshold_exceeded" in result["reasons"]
