import json
from pathlib import Path

import numpy as np

from ichor.core.adversarial.posterior import model_posterior_covariance
from ichor.core.models import Model


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ferebus_native_prefactor_golden.json"


def _write_golden_model(path: Path, payload):
    x_rows = [" ".join(map(str, row)) for row in payload["x_train"]]
    y_rows = [str(value) for value in payload["y_train"]]
    weight_rows = [str(value) for value in payload["weights"]]
    lines = [
        "# program FEREBUS",
        "# version 8.0.0",
        "# jitter " + str(payload["nugget"]),
        "",
        "[system]",
        "name GOLDEN",
        "atom O1",
        "property iqa",
        "ALF 1 2 3",
        "",
        "[dimensions]",
        "number_of_atoms 3",
        "number_of_features 2",
        "number_of_training_points 2",
        "",
        "[mean]",
        "type constant",
        "value " + str(payload["constant_mean"]),
        "",
        "[kernels]",
        "number_of_kernels 1",
        "composition k1",
        "prefactor " + str(payload["prefactor"]),
        "",
        "[kernel.k1]",
        "type rbf",
        "number_of_dimensions 2",
        "active_dimensions 1 2",
        "thetas " + " ".join(map(str, payload["thetas"])),
        "",
        "[training_data]",
        "units.x bohr bohr",
        "units.y Ha",
        "",
        "[training_data.x]",
        *x_rows,
        "",
        "[training_data.y]",
        *y_rows,
        "",
        "[weights]",
        *weight_rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def test_python_model_matches_native_ferebus_prefactor_golden(tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    model_path = tmp_path / "golden.model"
    _write_golden_model(model_path, payload)
    model = Model(model_path)

    assert model.kernel_prefactor == payload["prefactor"]
    assert np.allclose(model.R, payload["covariance"], rtol=1.0e-13, atol=1.0e-13)
    assert np.allclose(
        model.compute_weights().reshape(-1),
        payload["weights"],
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    query = np.asarray(payload["query"], dtype=float)
    assert np.allclose(
        model.r(query).reshape(-1),
        payload["query_cross_covariance"],
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.isclose(
        model.predict(query)[0],
        payload["prediction"],
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_log_marginal_likelihood_uses_two_sided_solve(tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    model_path = tmp_path / "golden.model"
    _write_golden_model(model_path, payload)
    model = Model(model_path)

    residual = model._y_minus_mean
    expected = (
        -0.5 * float((residual.T @ np.linalg.solve(model.R, residual)).item())
        - 0.5 * np.linalg.slogdet(model.R)[1]
        - 0.5 * model.ntrain * np.log(2.0 * np.pi)
    )
    assert np.isclose(model.compute_log_marginal_likelihood(), expected)


def test_posterior_prior_covariance_includes_native_prefactor(tmp_path):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    model_path = tmp_path / "golden.model"
    _write_golden_model(model_path, payload)
    model = Model(model_path)
    query = np.asarray(payload["query"], dtype=float).reshape(1, -1)

    posterior = model_posterior_covariance(model, query, query, scaled=False)
    cross = np.asarray(payload["query_cross_covariance"], dtype=float).reshape(-1, 1)
    expected = payload["prefactor"] - float(
        (cross.T @ np.linalg.solve(model.R, cross)).item()
    )
    assert np.isclose(posterior[0, 0], expected)
