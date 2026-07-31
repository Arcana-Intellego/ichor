import hashlib
from types import SimpleNamespace

from ichor.hpc.active_learning.versioning.trained_models import (
    TRAINED_MODEL_AUXILIARY_SUFFIXES,
    TRAINED_MODEL_DATASET_SPLITS,
    TrainedModelFile,
    TrainedModelSet,
    TrainedModelTask,
    TrainedModelVersioning,
    load_trained_models_from_snapshot,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_consumer_loader_hashes_and_parses_only_current_model_payloads(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    root = TrainedModelVersioning(
        campaign / "TRAINED_MODELS"
    ).iteration_path(2)
    model_path = root / "iqa" / "O1" / "WATER_iqa_O1.model"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("current model\n", encoding="utf-8")
    historical = campaign / "TRAINED_MODELS" / "iteration-000001" / "old.model"
    historical.parent.mkdir(parents=True)
    historical.write_text("historical model\n", encoding="utf-8")
    model_file = TrainedModelFile(
        relative_path="iqa/O1/WATER_iqa_O1.model",
        path=model_path,
        size=model_path.stat().st_size,
        sha256=_digest(model_path),
    )
    task = TrainedModelTask(
        task_index=1,
        property="iqa",
        atom="O1",
        alf_1_indexed=(1, 2, 3),
        directory="iqa/O1",
        model=model_file,
        config=model_file,
        execution_receipt=model_file,
        datasets={split: model_file for split in TRAINED_MODEL_DATASET_SPLITS},
        auxiliary={suffix: None for suffix in TRAINED_MODEL_AUXILIARY_SUFFIXES},
    )
    model_set = TrainedModelSet(
        version=2,
        campaign_uid="campaign-1",
        system="WATER",
        reference_data_version=2,
        reference_data_head_manifest_sha256="a" * 64,
        reference_data_view_sha256="b" * 64,
        parent_version=1,
        parent_manifest_sha256="c" * 64,
        properties=("iqa",),
        atoms=("O1",),
        tasks=(task,),
        root_files=(),
        model_set_sha256="d" * 64,
        evidence_set_sha256="e" * 64,
        head_manifest_sha256="f" * 64,
        root=root,
    )
    reference_view = SimpleNamespace(
        campaign_uid="campaign-1",
        head_manifest_sha256="a" * 64,
        cumulative_view_sha256="b" * 64,
    )
    snapshot = SimpleNamespace(
        reference_view=lambda version: reference_view,
        model_set=lambda version: model_set,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.versioning.reference_data.ReferenceDataVersioning.current_version",
        lambda _self: 2,
    )
    monkeypatch.setattr(
        "ichor.hpc.active_learning.versioning.trained_models.TrainedModelVersioning.current_version",
        lambda _self: 2,
    )
    hashed = []
    real_digest = _digest

    def _tracked_digest(path):
        hashed.append(path.resolve())
        return real_digest(path)

    monkeypatch.setattr(
        "ichor.hpc.active_learning.versioning.trained_models.sha256_file",
        _tracked_digest,
    )
    parsed = []
    sentinel = object()

    def _parse(root_path, model_paths):
        parsed.append((root_path, tuple(model_paths)))
        return sentinel

    monkeypatch.setattr(
        "ichor.core.models.Models.from_model_files",
        _parse,
    )

    observed_set, models, bindings = load_trained_models_from_snapshot(
        campaign,
        2,
        snapshot=snapshot,
        expected_campaign_uid="campaign-1",
    )

    assert observed_set is model_set
    assert models is sentinel
    assert hashed == [model_path.resolve()]
    assert parsed == [(root, (model_path,))]
    assert [binding.path for binding in bindings] == [model_path.resolve()]
    assert historical.resolve() not in hashed
