import json
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.hpc.active_learning.acquisition.gradient_diagnostics import (
    flatten_trace_gradient_diagnostics,
    static_gradient_diagnostics,
)
from ichor.hpc.active_learning.benchmark import acquisition_gradient as bench


class _FakeAtoms:
    coordinates = np.zeros((1, 3), dtype=float)

    def __len__(self):
        return 1

    def __iter__(self):
        return iter([SimpleNamespace(type="H")])


class _FakeAcquisition:
    def __init__(self, mode):
        self.config = SimpleNamespace(gradient=SimpleNamespace(mode=mode))
        self.mode_directions = [np.array([1.0, 0.0, 0.0])]
        self.posterior = SimpleNamespace(diagnostics={
            "n_means_batched_calls": 0,
            "n_covariance_matrix_batched_calls": 0,
        })

    def components(self, atoms, objective="full"):
        self.posterior.diagnostics["n_means_batched_calls"] += 1
        return SimpleNamespace(total=0.0)

    def gradient(self, atoms, mode=None, objective="full"):
        self.posterior.diagnostics["n_covariance_matrix_batched_calls"] += 1
        if mode == "active_fd":
            return np.array([[0.0, 1.0, 0.0]], dtype=float)
        return np.array([[1.0, 0.0, 0.0]], dtype=float)


def test_parse_gradient_modes_rejects_unknown_mode():
    assert bench._parse_gradient_modes("cartesian_fd, active_fd") == [
        "cartesian_fd",
        "active_fd",
    ]
    with pytest.raises(ValueError, match="gradient mode"):
        bench._parse_gradient_modes("cartesian_fd,unknown")


def test_static_gradient_diagnostics_handles_numpy_mode_directions():
    acquisition = SimpleNamespace(
        config=SimpleNamespace(gradient=SimpleNamespace(mode="active_fd")),
        mode_directions=np.eye(2),
        posterior=SimpleNamespace(diagnostics={"n_means_batched_calls": 3}),
    )

    payload = static_gradient_diagnostics(acquisition, _FakeAtoms())

    assert payload["gradient_mode"] == "active_fd"
    assert payload["natoms"] == 1
    assert payload["subspace_dim"] == 2
    assert payload["n_estimated_acquisition_value_calls_per_gradient"] == 4
    assert payload["posterior_diagnostics"]["n_means_batched_calls"] == 3


def test_flatten_trace_gradient_diagnostics_keeps_compact_scalar_fields():
    payload = flatten_trace_gradient_diagnostics({
        "gradient_mode": "active_fd",
        "gradient_call_count": 2,
        "gradient_wall_seconds_total": 1.5,
        "posterior_diagnostics": {
            "n_means_batched_calls": 4,
            "ignored": 99,
        },
    })

    assert payload == {
        "gradient_mode": "active_fd",
        "gradient_call_count": 2,
        "gradient_wall_seconds_total": 1.5,
        "posterior_n_means_batched_calls": 4,
    }


def test_run_benchmark_uses_context_and_reports_mode_comparison(monkeypatch, tmp_path):
    fake_context = {
        "campaign": tmp_path,
        "config": SimpleNamespace(),
        "pool": SimpleNamespace(),
        "state": SimpleNamespace(models_version=7),
        "models": object(),
        "iter_dir": tmp_path / "7_ACTIVE_LEARNING" / "iteration-0000",
        "seed_frame_id": 12,
        "seed_atoms": _FakeAtoms(),
        "trajectory": [_FakeAtoms()],
        "reference_scales": None,
    }
    monkeypatch.setattr(
        bench,
        "_load_context",
        lambda campaign_dir, iteration, seed_index: fake_context,
    )
    monkeypatch.setattr(
        bench,
        "_build_acquisition",
        lambda context, mode, **kwargs: _FakeAcquisition(mode),
    )

    payload = bench.run_benchmark(
        campaign_dir=tmp_path,
        iteration=0,
        seed_index=0,
        gradient_modes=("cartesian_fd", "active_fd"),
        repeat=2,
    )

    assert payload["schema_version"] == 1
    assert payload["models_version"] == 7
    assert payload["seed_frame_id"] == 12
    assert len(payload["runs"]) == 4
    assert payload["gradient_backend"] == "direct"
    assert payload["objective"] == "full"
    assert {
        row["gradient_mode"] for row in payload["runs"]
    } == {"cartesian_fd", "active_fd"}
    comparison = payload["comparisons"]["active_fd_vs_cartesian_fd"]
    assert comparison["cosine"] == 0.0
    assert comparison["speedup"] is not None
    json.dumps(bench._json_safe(payload))


def test_main_writes_json_payload(monkeypatch, tmp_path, capsys):
    output = tmp_path / "bench.json"
    captured = {}

    def fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "runs": [{
                "gradient_mode": "active_fd",
                "repeat": 0,
                "wall_seconds": 0.25,
                "grad_norm": 1.0,
                "n_estimated_acquisition_value_calls": 2,
                "gradient_backend": kwargs["gradient_backend"],
                "objective": kwargs["objective"],
            }],
            "comparisons": {},
        }

    monkeypatch.setattr(
        bench,
        "run_benchmark",
        fake_run_benchmark,
    )

    rc = bench.main([
        "--campaign-dir",
        str(tmp_path),
        "--iteration",
        "0",
        "--seed-index",
        "0",
        "--gradient-mode",
        "active_fd",
        "--gradient-backend",
        "process",
        "--objective",
        "cheap_driver",
        "--driver-gradient-backend",
        "hybrid_geometry",
        "--workers",
        "3",
        "--json",
        str(output),
    ])

    assert rc == 0
    assert captured["gradient_backend"] == "process"
    assert captured["objective"] == "cheap_driver"
    assert captured["driver_gradient_backend"] == "hybrid_geometry"
    assert captured["workers"] == 3
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert "active_fd" in capsys.readouterr().out
