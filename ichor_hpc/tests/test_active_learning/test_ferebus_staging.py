"""FEREBUS staging orchestration with the cluster-only PointsDirectory export stubbed.

The daemon stages pyferebus's flat property directories, a pyferebus job-details file, and a
strict daemon manifest. pyferebus itself later moves the CSVs into per-atom datasets folders.
"""
import json

import ichor.core.files as core_files

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.versioning.manifest import ManifestMismatchError, write_manifest


class _FakePointsDirectory:
    """stand-in for the real PointsDirectory: instead of reading AIMAll .int files it just emits a
    known per-atom feature/iqa csv (the iqa scale differs per element, as it really would)."""
    def __init__(self, path):
        self._path = path

    def __iter__(self):
        for i in range(20):
            yield self._path / ("POINT_" + str(i).zfill(4) + ".pointdir")

    def alf_dict(self, _calc):
        return {"O1": (0, 1, 2), "H2": (1, 0, 2), "H3": (2, 0, 1)}

    def features_with_properties_to_csv(self, _alf, str_to_append_to_fname="_train.csv",
                                        property_types=None):
        # stage_ferebus_inputs has chdir'd into the staging dir, so a bare filename lands there.
        for atom, base in (("O1", -75.0), ("H2", -0.5), ("H3", -0.5)):
            with open(atom + str_to_append_to_fname, "w", encoding="utf-8", newline="\n") as f:
                f.write("f1,f2,f3,iqa,q00\n")
                for i in range(20):
                    f.write(
                        f"{i + 0.1},{i + 0.2},{i + 0.3},"
                        f"{base - i * 0.01},{0.2 + i * 0.001}\n"
                    )


def test_stage_ferebus_inputs_orchestration(tmp_path, monkeypatch):
    campaign = tmp_path / "c"
    # the (real) is_dir() guard needs a committed training dir to exist.
    training_dir = campaign / "5_TRAINING" / "iteration-0000"
    training_dir.mkdir(parents=True)
    write_manifest(training_dir, {})
    # stub only the cluster-only export. the function does `from ichor.core.files import
    # PointsDirectory` at call time, so patching the module attribute takes effect.
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    cfg = CampaignConfig()
    cfg.system_name = "WATER"
    cfg.ferebus.properties = ["iqa", "q00"]
    staging, n_tasks = stg.stage_ferebus_inputs(campaign, cfg, training_version=0, is_initial=False)

    assert n_tasks == 6
    assert set((staging / "ATOMS.txt").read_text(encoding="utf-8").split()) == {"O1", "H2", "H3"}
    assert b"\r" not in (staging / "ATOMS.txt").read_bytes()
    assert (staging / "PROPERTIES.txt").read_text(encoding="utf-8").split() == ["iqa", "q00"]
    for prop in ("iqa", "q00"):
        for atom in ("O1", "H2", "H3"):
            for kind in ("TRAINING", "INT_VALIDATION", "EXT_VALIDATION"):
                p = staging / prop / f"WATER_{atom}_{kind}_SET.csv"
                assert p.is_file()
                assert p.read_text(encoding="utf-8").splitlines()[0] == "f1,f2,f3,iqa,q00"
    jd = (staging / stg.FEREBUS_JOB_DETAILS).read_text(encoding="utf-8")
    assert "props iqa q00" in jd
    assert "O1 1 2 3" in jd
    assert "q00-O1 " in jd
    manifest = json.loads((staging / stg.FEREBUS_TASK_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["n_tasks"] == 6
    assert manifest["pointdir_row_order"] == [
        "POINT_" + str(i).zfill(4) + ".pointdir" for i in range(20)
    ]
    assert manifest["degenerate_property_stats"] == []
    assert {task["property"] for task in manifest["tasks"]} == {"iqa", "q00"}
    assert {tuple(task["alf_1_indexed"]) for task in manifest["tasks"]} >= {(1, 2, 3)}
    assert not list(staging.glob("ferebus_*.toml"))
    # the transient scratch csvs do not survive into the staging dir
    assert not list(staging.glob("*_normalised_for_split.csv"))


def test_stage_ferebus_inputs_clears_stale_models(tmp_path, monkeypatch):
    # A14/A15: a stale .model / *_train.csv from a previous iteration in the shared staging dir must
    # be wiped before this run, not globbed back in.
    campaign = tmp_path / "c"
    training_dir = campaign / "5_TRAINING" / "iteration-0000"
    training_dir.mkdir(parents=True)
    write_manifest(training_dir, {})
    staging = campaign / "6_TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    (staging / "STALE.model").write_text("old model from a previous iteration", encoding="utf-8")
    (staging / "Zz9_train.csv").write_text("f1,iqa\n0.1,-1.0\n", encoding="utf-8")
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    cfg = CampaignConfig()
    cfg.system_name = "WATER"
    stg.stage_ferebus_inputs(campaign, cfg, training_version=0, is_initial=False)

    assert not (staging / "STALE.model").exists()
    assert not (staging / "Zz9_train.csv").exists()


def test_stage_ferebus_inputs_rejects_unmanifested_committed_pointdir(tmp_path, monkeypatch):
    campaign = tmp_path / "c"
    training_dir = campaign / "5_TRAINING" / "iteration-0000"
    rogue = training_dir / "POINT_9999.pointdir"
    rogue.mkdir(parents=True)
    (rogue / "input.gjf").write_text("%chk=x\n", encoding="utf-8")
    write_manifest(training_dir, {})
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    cfg = CampaignConfig()
    cfg.system_name = "WATER"
    try:
        stg.stage_ferebus_inputs(campaign, cfg, training_version=0, is_initial=False)
    except ManifestMismatchError:
        pass
    else:
        raise AssertionError("unmanifested committed pointdir was not rejected")
