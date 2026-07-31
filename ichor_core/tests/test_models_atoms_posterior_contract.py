import numpy as np
import pytest

from ichor.core.adversarial.posterior import TotalEnergyPosterior
from ichor.core.atoms import Atom, Atoms
from ichor.core.models import Model, Models


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
        "prefactor 1.0",
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


def test_production_model_cholesky_and_identity_are_stable(tmp_path):
    model = Model(_model_dir(tmp_path) / "WATER_iqa_O1.model")

    identity = model.numeric_identity
    first = model.lower_cholesky
    second = model.lower_cholesky

    assert len(identity) == 64
    assert model.numeric_identity == identity
    assert first is second
    assert not first.flags.writeable


def test_production_model_accepts_only_identity_bound_cholesky_factor(tmp_path):
    model = Model(_model_dir(tmp_path) / "WATER_iqa_O1.model")
    expected = np.linalg.cholesky(model.R)

    model.invalidate_numeric_cache()
    model.install_lower_cholesky(
        expected,
        expected_numeric_identity=model.numeric_identity,
    )

    assert model.lower_cholesky is expected
    assert not model.lower_cholesky.flags.writeable
    with pytest.raises(ValueError, match="identity"):
        model.install_lower_cholesky(
            expected,
            expected_numeric_identity="0" * 64,
        )


def test_total_energy_posterior_rejects_non_finite_feature_values(tmp_path):
    posterior = TotalEnergyPosterior(Models(_model_dir(tmp_path)))
    frame = _atoms()
    frame[1].coordinates[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        posterior.variance(frame)


def test_models_exact_loader_uses_only_manifest_selected_nested_files(tmp_path):
    selected = tmp_path / "iqa" / "O1" / "WATER_iqa_O1.model"
    selected.parent.mkdir(parents=True)
    _write_model(selected, atom="O1", alf=(1, 2, 3))
    unlisted = tmp_path / "iqa" / "H2" / "WATER_iqa_H2.model"
    unlisted.parent.mkdir(parents=True)
    _write_model(unlisted, atom="H2", alf=(2, 1, 3))

    models = Models.from_model_files(
        tmp_path,
        ["iqa/O1/WATER_iqa_O1.model"],
    )

    assert [(model.type, model.atom) for model in models] == [("iqa", "O1")]


def test_models_exact_loader_rejects_path_outside_root(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside.model")
    _write_model(outside, atom="O1", alf=(1, 2, 3))
    with pytest.raises(ValueError, match="escapes the model root"):
        Models.from_model_files(tmp_path, [outside])


def test_models_exact_loader_rejects_duplicate_paths(tmp_path):
    model = tmp_path / "WATER_iqa_O1.model"
    _write_model(model, atom="O1", alf=(1, 2, 3))
    with pytest.raises(ValueError, match="duplicate explicit model path"):
        Models.from_model_files(tmp_path, [model, model])


def test_model_parser_rejects_nonblank_content_after_exact_weights(tmp_path):
    path = tmp_path / "WATER_iqa_O1.model"
    _write_model(path, atom="O1", alf=(1, 2, 3))
    path.write_text(
        path.read_text(encoding="utf-8") + "unexpected trailing token\n",
        encoding="utf-8",
        newline="\n",
    )

    model = Model(path)
    with pytest.raises(ValueError, match="unexpected content after FEREBUS weights"):
        _ = model.weights


def test_model_parser_requires_explicit_kernel_prefactor(tmp_path):
    path = tmp_path / "WATER_iqa_O1.model"
    _write_model(path, atom="O1", alf=(1, 2, 3))
    path.write_text(
        path.read_text(encoding="utf-8").replace("prefactor 1.0\n", ""),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="missing kernel prefactor"):
        _ = Model(path).weights


def test_model_parser_requires_exact_kernel_section_count(tmp_path):
    path = tmp_path / "WATER_iqa_O1.model"
    _write_model(path, atom="O1", alf=(1, 2, 3))
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "number_of_kernels 1",
            "number_of_kernels 2",
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="kernel section count mismatch"):
        _ = Model(path).weights


def test_model_parser_rejects_unreferenced_kernel_sections(tmp_path):
    path = tmp_path / "WATER_iqa_O1.model"
    _write_model(path, atom="O1", alf=(1, 2, 3))
    text = path.read_text(encoding="utf-8")
    extra = (
        "[kernel.k2]\n"
        "type rbf\n"
        "number_of_dimensions 3\n"
        "active_dimensions 1 2 3\n"
        "thetas 1.0 1.0 1.0\n\n"
    )
    text = text.replace("number_of_kernels 1", "number_of_kernels 2")
    text = text.replace("[training_data]\n", extra + "[training_data]\n")
    path.write_text(text, encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="composition/section coverage mismatch"):
        _ = Model(path).weights


def test_models_atom_names_are_canonical_and_stably_deduplicated(tmp_path):
    paths = []
    for filename, atom, alf in (
        ("z_H3.model", "H3", (3, 1, 2)),
        ("a_O1.model", "O1", (1, 2, 3)),
        ("m_H2.model", "H2", (2, 1, 3)),
        ("n_q00_H2.model", "H2", (2, 1, 3)),
    ):
        path = tmp_path / filename
        _write_model(path, atom=atom, alf=alf)
        paths.append(path)

    models = Models.from_model_files(tmp_path, paths)

    assert models.atom_names == ["O1", "H2", "H3"]
