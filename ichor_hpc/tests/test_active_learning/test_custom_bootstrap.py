from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.custom_bootstrap import (
    BootstrapInputError,
    commit_bootstrap_plan,
    inspect_bootstrap_inputs,
    read_custom_bootstrap_manifest,
)
from ichor.hpc.active_learning.versioning.manifest import sha256_file


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"


def _pool_frames():
    from ichor.core.files.xyz import Trajectory

    trajectory = Trajectory(FIXTURE)
    trajectory.read()
    return [frame.copy() for frame in trajectory]


def _write_xyz(path: Path, frames) -> None:
    lines = []
    for index, frame in enumerate(frames):
        lines.extend((str(len(frame)), "bootstrap " + str(index)))
        lines.extend(
            str(atom.type)
            + " "
            + format(float(atom.x), ".16g")
            + " "
            + format(float(atom.y), ".16g")
            + " "
            + format(float(atom.z), ".16g")
            for atom in frame
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write_csv(path: Path, frames, *, extras=("iqa", "q00")) -> list[int]:
    from ichor.core.calculators import (
        calculate_alf_atom_sequence,
        calculate_alf_features,
    )

    alf = calculate_alf_atom_sequence(frames[0][0])
    features = [
        np.asarray(calculate_alf_features(frame[0], alf), dtype=float)
        for frame in frames
    ]
    header = ["f" + str(index) for index in range(1, features[0].size + 1)]
    header.extend(str(value) for value in extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row_index, row in enumerate(features):
            writer.writerow(
                [format(float(value), ".17g") for value in row]
                + [format(float(row_index + offset), ".17g") for offset in range(len(extras))]
            )
    values = [int(alf.origin_idx) + 1, int(alf.x_axis_idx) + 1]
    if alf.xy_plane_idx is not None:
        values.append(int(alf.xy_plane_idx) + 1)
    return values


def _inspect(campaign: Path, config: CampaignConfig, frames, **kwargs):
    return inspect_bootstrap_inputs(
        campaign,
        config,
        frames,
        pool_sha256=sha256_file(campaign / "pool.xyz"),
        **kwargs,
    )


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_mixed_xyz_and_csv_splits_are_accepted_and_csv_extras_are_recorded(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    bootstrap = tmp_path / "bootstrap"
    _write_xyz(bootstrap / "training_set_bootstrap.xyz", [frames[0]])
    alf = _write_csv(
        bootstrap / "internal_validation_set_bootstrap.csv",
        [frames[1]],
    )
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True
    prompts = []

    plan = _inspect(
        tmp_path,
        config,
        frames,
        alf_prompt=lambda split, path, atoms: prompts.append((split, path.name)) or alf,
    )

    assert prompts == [("int_val", "internal_validation_set_bootstrap.csv")]
    assert plan.sources["train"].kind == "xyz"
    assert plan.sources["int_val"].kind == "csv"
    assert plan.sources["int_val"].extra_columns == ("iqa", "q00")
    assert plan.polus_deficits == {"train": 7, "int_val": 1, "ext_val": 2}


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_duplicate_format_for_one_split_is_rejected(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    bootstrap = tmp_path / "bootstrap"
    _write_xyz(bootstrap / "external_validation_set_bootstrap.xyz", [frames[0]])
    _write_csv(bootstrap / "external_validation_set_bootstrap.csv", [frames[1]])
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True

    with pytest.raises(BootstrapInputError, match="both XYZ and CSV"):
        _inspect(tmp_path, config, frames, alf_values={"ext_val": [1, 2, 3]})


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_confirmed_bootstrap_source_change_is_rejected_before_commit(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    source = tmp_path / "bootstrap" / "training_set_bootstrap.xyz"
    _write_xyz(source, [frames[0]])
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True
    plan = _inspect(tmp_path, config, frames)

    _write_xyz(source, [frames[1]])

    with pytest.raises(BootstrapInputError, match="changed after inspection"):
        commit_bootstrap_plan(plan)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_model_directory_and_training_split_source_are_rejected(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    bootstrap = tmp_path / "bootstrap"
    (bootstrap / "model_krig").mkdir(parents=True)
    _write_xyz(bootstrap / "training_set_bootstrap.xyz", [frames[0]])
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True

    with pytest.raises(BootstrapInputError, match="both supply training data"):
        _inspect(tmp_path, config, frames)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_model_directory_rejects_non_model_files(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    model_dir = tmp_path / "bootstrap" / "model_krig"
    model_dir.mkdir(parents=True)
    (model_dir / "stale.csv").write_text("not a model\n", encoding="utf-8")
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True

    with pytest.raises(BootstrapInputError, match="non-model file"):
        _inspect(tmp_path, config, frames)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_custom_bootstrap_flag_and_input_presence_must_agree(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True
    with pytest.raises(BootstrapInputError, match="no recognised inputs"):
        _inspect(tmp_path, config, frames)

    config.campaign.custom_bootstrap = False
    _write_xyz(
        tmp_path / "bootstrap" / "training_set_bootstrap.xyz",
        [frames[0]],
    )
    with pytest.raises(BootstrapInputError, match="custom_bootstrap is false"):
        _inspect(tmp_path, config, frames)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_split_source_cannot_exceed_configured_target(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    _write_xyz(
        tmp_path / "bootstrap" / "internal_validation_set_bootstrap.xyz",
        frames[:3],
    )
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True

    with pytest.raises(BootstrapInputError, match="permits at most 2"):
        _inspect(tmp_path, config, frames)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_numerically_identical_bootstrap_geometry_excludes_pool_frame(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    rounded = frames[0].copy()
    rounded[0].coordinates[0] += 5.0e-7
    _write_xyz(
        tmp_path / "bootstrap" / "training_set_bootstrap.xyz",
        [rounded],
    )
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True

    plan = _inspect(tmp_path, config, frames)

    assert 0 in plan.excluded_pool_frame_ids


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_committed_bootstrap_reader_rejects_tampered_canonical_geometry(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    _write_xyz(
        tmp_path / "bootstrap" / "training_set_bootstrap.xyz",
        [frames[0]],
    )
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True
    payload = commit_bootstrap_plan(_inspect(tmp_path, config, frames))
    canonical = (
        tmp_path
        / str(payload["bootstrap_inputs_root"])
        / str(payload["sources"]["train"]["canonical_xyz"])
    )
    canonical.write_text(
        canonical.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BootstrapInputError, match="canonical bootstrap SHA mismatch"):
        read_custom_bootstrap_manifest(tmp_path)


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_bootstrap_commit_repairs_pointer_after_rename_crash_window(tmp_path):
    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    config = CampaignConfig()
    plan = _inspect(tmp_path, config, frames)
    committed = commit_bootstrap_plan(plan)
    pointer = tmp_path / ".DATA" / "ACTIVE_LEARNING" / "CUSTOM_BOOTSTRAP.json"
    pointer.unlink()

    repaired = commit_bootstrap_plan(plan)

    assert pointer.is_file()
    assert repaired["plan_identity_sha256"] == committed["plan_identity_sha256"]


@pytest.mark.skipif(not FIXTURE.is_file(), reason="water fixture missing")
def test_model_bootstrap_requires_every_configured_property_and_atom(tmp_path, monkeypatch):
    from ichor.core.calculators import (
        calculate_alf_atom_sequence,
        calculate_alf_features,
    )
    import ichor.core.models as models_module

    shutil.copy2(FIXTURE, tmp_path / "pool.xyz")
    frames = _pool_frames()
    model_dir = tmp_path / "bootstrap" / "model_krig"
    model_dir.mkdir(parents=True)
    for atom in frames[0].atom_names:
        (model_dir / ("iqa_" + str(atom) + ".model")).write_text(
            "fixture\n", encoding="utf-8"
        )

    class FakeModel:
        def __init__(self, path):
            stem = Path(path).stem
            self.type, self.atom = stem.split("_", 1)
            self.system_name = "SYSTEM"
            atom_index = frames[0].atom_names.index(self.atom)
            alf = calculate_alf_atom_sequence(frames[0][atom_index])
            self.ialf = np.asarray(list(alf), dtype=int)
            self.x = np.asarray(
                [calculate_alf_features(frame[atom_index], alf) for frame in frames[:2]],
                dtype=float,
            )
            self.y = np.zeros(2, dtype=float)
            self.weights = np.zeros(2, dtype=float)
            self.ntrain = 2
            self.nfeats = self.x.shape[1]

    monkeypatch.setattr(models_module, "Model", FakeModel)
    config = CampaignConfig()
    config.campaign.custom_bootstrap = True
    config.ferebus.properties = ["iqa", "q00"]

    with pytest.raises(BootstrapInputError, match="coverage mismatch") as error:
        _inspect(tmp_path, config, frames)
    assert "q00" in str(error.value)
