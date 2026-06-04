import numpy as np
import pytest

from ichor.core.adversarial.posterior import TotalEnergyPosterior
from ichor.core.atoms import Atom, Atoms
from ichor.core.models import Models


def _write_model(path, *, atom, alf, ntrain=5, nfeats=3):
    rows = [
        [0.2 + i * 0.1 + j * 0.02 for j in range(nfeats)]
        for i in range(ntrain)
    ]
    lines = [
        "# jitter 1.0e-6",
        "# likelihood -1.0",
        "",
        "[system]",
        "name WATER",
        "atom " + atom,
        "property iqa",
        "ALF " + " ".join(str(int(x)) for x in alf),
        "",
        "[dimensions]",
        "number_of_atoms 3",
        "number_of_features " + str(nfeats),
        "number_of_training_points " + str(ntrain),
        "",
        "[mean]",
        "type zero",
        "",
        "[kernels]",
        "number_of_kernels 1",
        "composition k1",
        "",
        "[kernel.k1]",
        "type rbf",
        "number_of_dimensions " + str(nfeats),
        "active_dimensions " + " ".join(str(i + 1) for i in range(nfeats)),
        "thetas " + " ".join("1.0" for _ in range(nfeats)),
        "",
        "[training_data]",
        "units.x bohr bohr radians",
        "units.y Ha",
        "",
        "[training_data.x]",
    ]
    lines += [" ".join(str(v) for v in row) for row in rows]
    lines += ["", "[training_data.y]"]
    lines += [str(-1.0 - i * 0.1) for i in range(ntrain)]
    lines += ["", "[weights]"]
    lines += ["0.0" for _ in range(ntrain)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _atoms():
    return Atoms([
        Atom("O", 0.0, 0.0, 0.0),
        Atom("H", 0.96, 0.0, 0.0),
        Atom("H", 0.0, 0.96, 0.0),
    ])


def _model_dir(tmp_path):
    specs = {
        "O1": (1, 2, 3),
        "H2": (2, 1, 3),
        "H3": (3, 1, 2),
    }
    for atom, alf in specs.items():
        _write_model(tmp_path / ("WATER_iqa_" + atom + ".model"), atom=atom, alf=alf)
    return tmp_path


def test_models_features_from_atoms_uses_atom_names_for_alf_lookup(tmp_path):
    models = Models(_model_dir(tmp_path))
    features = models._features_from_atoms(_atoms())
    assert set(features) == {"O1", "H2", "H3"}
    for value in features.values():
        assert value.shape == (3,)
        assert np.all(np.isfinite(value))


def test_total_energy_posterior_accepts_atoms_frames_with_real_models(tmp_path):
    posterior = TotalEnergyPosterior(Models(_model_dir(tmp_path)))
    frame = _atoms()
    variance = posterior.variance(frame)
    variances = posterior.variances([frame, frame.copy()])
    assert np.isfinite(variance)
    assert variances.shape == (2,)
    assert np.all(np.isfinite(variances))


def test_total_energy_posterior_rejects_non_finite_feature_values(tmp_path):
    posterior = TotalEnergyPosterior(Models(_model_dir(tmp_path)))
    frame = _atoms()
    frame[1].coordinates[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        posterior.variance(frame)
