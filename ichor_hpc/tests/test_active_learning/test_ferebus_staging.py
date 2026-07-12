"""FEREBUS staging orchestration with the cluster-only PointsDirectory export stubbed.

The daemon stages pyferebus's flat property directories, a pyferebus job-details file, and a
strict daemon manifest. pyferebus itself later moves the CSVs into per-atom datasets folders.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import ichor.core.files as core_files
import numpy as np

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon import input_staging as stg
from ichor.hpc.active_learning.point_allocation import (
    accepted_attempts,
    allocation_targets,
    create_point_allocation,
    pending_attempts,
    point_allocation_path,
    record_quantum_results,
)
from ichor.hpc.active_learning.versioning.manifest import (
    ManifestMismatchError,
    compute_directory_manifest,
    write_manifest,
)
from ichor.hpc.active_learning.versioning.provenance import (
    enrich_with_point_allocation,
    write_seed_provenance,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "live_outputs"


class _FakePointsDirectory(list):
    """stand-in for the real PointsDirectory: instead of reading AIMAll .int files it just emits a
    known per-atom feature/iqa csv (the iqa scale differs per element, as it really would)."""
    def __init__(self, path, needs_parsing=True):
        super().__init__()
        self._path = path
        self.path = path

    def alf_dict(self, _calc):
        return {"O1": (0, 1, 2), "H2": (1, 0, 2), "H3": (2, 0, 1)}

    def features_with_properties_to_csv(self, _alf, str_to_append_to_fname="_train.csv",
                                        property_types=None):
        # stage_ferebus_inputs has chdir'd into the staging dir, so a bare filename lands there.
        properties = [str(value) for value in (property_types or ["iqa"])]
        for atom, base in (("O1", -75.0), ("H2", -0.5), ("H3", -0.5)):
            with open(atom + str_to_append_to_fname, "w", encoding="utf-8", newline="\n") as f:
                f.write("f1,f2,f3," + ",".join(properties) + "\n")
                for i in range(len(self)):
                    values = {
                        "iqa": base - i * 0.01,
                        "q00": 0.2 + i * 0.001,
                    }
                    f.write(
                        f"{i + 0.1},{i + 0.2},{i + 0.3},"
                        + ",".join(str(values[prop]) for prop in properties)
                        + "\n"
                    )


def _prepare_bootstrap_training(
    campaign,
    cfg,
    *,
    n_train=12,
    n_int_val=3,
    n_ext_val=5,
):
    source_dir = campaign / ".DATA" / "STAGING" / "initial"
    if int(n_train) > 0:
        cfg.point_allocation.bootstrap_training_size = int(n_train)
    cfg.point_allocation.bootstrap_internal_validation_size = int(n_int_val)
    cfg.point_allocation.bootstrap_external_validation_size = int(n_ext_val)
    n_total = int(n_train) + int(n_int_val) + int(n_ext_val)
    targets = {
        "train": int(n_train),
        "int_val": int(n_int_val),
        "ext_val": int(n_ext_val),
        "total": int(n_total),
    }
    candidates = [
        {
            "candidate_id": "candidate-" + str(i),
            "frame_id": i,
            "pointdir_name": "POINT_" + str(i).zfill(4) + ".pointdir",
        }
        for i in range(n_total)
    ]
    allocation_path = point_allocation_path(
        campaign,
        context="bootstrap",
        iteration=0,
    )
    allocation = create_point_allocation(
        allocation_path,
        campaign_uid="campaign-uid",
        context="bootstrap",
        iteration=0,
        targets=(allocation_targets(cfg, "bootstrap") if int(n_train) > 0 else targets),
        primary_candidates=candidates,
        reserve_candidates=[],
    )
    results = []
    for attempt in pending_attempts(allocation):
        pointdir = source_dir / str(attempt["pointdir_name"])
        pointdir.mkdir(parents=True)
        (pointdir / "input.gjf").write_text("# synthetic\n", encoding="utf-8")
        results.append({
            "candidate_id": attempt["candidate_id"],
            "accepted": True,
            "pointdir": str(pointdir),
        })
    allocation = record_quantum_results(allocation_path, results)
    for attempt in accepted_attempts(allocation):
        pointdir = source_dir / str(attempt["pointdir_name"])
        write_seed_provenance(
            pointdir,
            campaign_uid="campaign-uid",
            iteration=0,
            trajectory_sha256="a" * 64,
            seed_frame_id=int(attempt["frame_id"]),
            seed_selection_origin="bootstrap",
            seed_variance_at_selection=None,
            subspace_neighbour_frame_ids=[],
            subspace_dimension=0,
            subspace_eigenvalues=[],
        )
        enrich_with_point_allocation(
            pointdir,
            candidate_id=str(attempt["candidate_id"]),
            context="bootstrap",
            slot_id=int(attempt["slot_id"]),
            split=str(attempt["split"]),
            allocation_slot_assignment_sha256=str(
                allocation["slot_assignment_sha256"]
            ),
        )
    stg.commit_reference_data_delta(
        campaign,
        reference_data_version=0,
        context="bootstrap",
        iteration=0,
    )
    return campaign / "QM_REFERENCE_DATA" / "iteration-000000"


def test_stage_ferebus_inputs_orchestration(tmp_path, monkeypatch):
    campaign = tmp_path / "c"
    # stub only the cluster-only export. the function does `from ichor.core.files import
    # PointsDirectory` at call time, so patching the module attribute takes effect.
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    cfg = CampaignConfig()
    cfg.system_name = "WATER"
    cfg.ferebus.properties = ["iqa", "q00"]
    _prepare_bootstrap_training(campaign, cfg)
    staging, n_tasks = stg.stage_ferebus_inputs(
        campaign,
        cfg,
        reference_data_version=0,
        is_initial=False,
    )

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
    assert stg.read_ferebus_manifest(staging) == manifest
    assert manifest["n_tasks"] == 6
    assert manifest["split_ledger"]["counts"] == {"train": 12, "int_val": 3, "ext_val": 5}
    assert manifest["pointdir_row_order"] == [
        "POINT_" + str(i).zfill(6) + ".pointdir" for i in range(20)
    ]
    assert manifest["degenerate_property_stats"] == []
    assert {task["property"] for task in manifest["tasks"]} == {"iqa", "q00"}
    assert {
        tuple(task["row_counts"][key] for key in ("train", "int_val", "ext_val"))
        for task in manifest["tasks"]
    } == {(12, 3, 5)}
    assert {tuple(task["alf_1_indexed"]) for task in manifest["tasks"]} >= {(1, 2, 3)}
    for task in manifest["tasks"]:
        assert set(task["datasets"]) == {"train", "int_val", "ext_val"}
        for split, expected_rows in (("train", 12), ("int_val", 3), ("ext_val", 5)):
            record = task["datasets"][split]
            source = staging / task["property"] / Path(record["path"]).name
            assert record["rows"] == expected_rows
            assert record["size"] == source.stat().st_size
            assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not list(staging.glob("ferebus_*.toml"))
    # the transient scratch csvs do not survive into the staging dir
    assert not list(staging.glob("*_normalised_for_split.csv"))


def test_model_bootstrap_stages_exact_historical_training_prefix(
    tmp_path,
    monkeypatch,
):
    from ichor.core.calculators import (
        calculate_alf_atom_sequence,
        calculate_alf_features,
    )
    from ichor.core.files.xyz import Trajectory
    from ichor.core.models import Model
    from ichor.hpc.active_learning.custom_bootstrap import (
        commit_bootstrap_plan,
        inspect_bootstrap_inputs,
    )
    from ichor.hpc.active_learning.daemon.dry_run_executor import (
        DryRunPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.live_executor import (
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import fresh_campaign_state

    campaign = tmp_path / "c"
    pool_frames = []
    for xyz_path in sorted((FIXTURES / "initial_quantum").glob("*.pointdir/*.xyz")):
        trajectory = Trajectory(xyz_path)
        trajectory.read()
        pool_frames.extend(frame.copy() for frame in trajectory)
    assert len(pool_frames) >= 4
    pool_frames = pool_frames[:4]

    cfg = CampaignConfig()
    cfg.campaign.system_name = "WATER"
    cfg.campaign.custom_bootstrap = True
    cfg.ferebus.properties = ["iqa"]
    cfg.point_allocation.bootstrap_training_size = 8
    cfg.point_allocation.bootstrap_internal_validation_size = 2
    cfg.point_allocation.bootstrap_external_validation_size = 2
    model_frames = [pool_frames[0].copy(), pool_frames[1].copy()]
    model_frames[0][1].coordinates[0] += 0.015
    model_frames[1][2].coordinates[1] -= 0.020
    model_dir = campaign / "bootstrap" / "model_krig"
    model_dir.mkdir(parents=True)
    baseline_by_atom = {}
    for atom_index, atom_name in enumerate(model_frames[0].atom_names):
        alf = calculate_alf_atom_sequence(model_frames[0][atom_index])
        features = np.asarray(
            [
                calculate_alf_features(frame[atom_index], alf)
                for frame in model_frames
            ],
            dtype=float,
        )
        targets = np.asarray(
            [-75.0 - atom_index * 0.1, -75.01 - atom_index * 0.1],
            dtype=float,
        )
        model_path = model_dir / ("WATER_iqa_" + atom_name + ".model")
        DryRunPhaseExecutor._write_dry_ferebus_model(
            model_path,
            system="WATER",
            atom=atom_name,
            prop="iqa",
            alf_1_indexed=[int(value) + 1 for value in alf],
            ntrain=2,
            feature_rows=features,
            target_rows=targets,
        )
        baseline_by_atom[atom_name] = (features, targets)

    plan = inspect_bootstrap_inputs(
        campaign,
        cfg,
        pool_frames,
        pool_sha256="a" * 64,
    )
    assert plan.model is not None
    assert plan.model.training_count == 2
    assert plan.polus_deficits == {"train": 0, "int_val": 2, "ext_val": 2}
    commit_bootstrap_plan(plan)
    _prepare_bootstrap_training(
        campaign,
        cfg,
        n_train=0,
        n_int_val=2,
        n_ext_val=2,
    )
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    staging, n_tasks = stg.stage_ferebus_inputs(
        campaign,
        cfg,
        reference_data_version=0,
        is_initial=False,
    )
    manifest = stg.prepare_imported_model_bootstrap(staging)

    assert n_tasks == 3
    assert manifest["model_bootstrap"]["historical_training_rows"] == 2
    for task in manifest["tasks"]:
        assert task["row_counts"] == {"train": 2, "int_val": 2, "ext_val": 2}
        model = Model(
            stg.resolve_ferebus_task_path(
                staging,
                task["expected_model_path"],
                "expected_model_path",
            )
        )
        expected_x, expected_y = baseline_by_atom[str(task["atom"])]
        assert np.allclose(np.asarray(model.x, dtype=float), expected_x)
        assert np.allclose(np.asarray(model.y, dtype=float).reshape(-1), expected_y)
        training_csv = stg.resolve_ferebus_task_path(
            staging,
            task["training_csv"],
            "training_csv",
        )
        with training_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert np.allclose(
            [[float(row["f1"]), float(row["f2"]), float(row["f3"])] for row in rows],
            expected_x,
        )
        assert np.allclose([float(row["iqa"]) for row in rows], expected_y)

    executor = DryRunPhaseExecutor(campaign_dir=campaign, config=cfg)
    state = fresh_campaign_state(campaign_uid="campaign-uid")
    state.reference_data_version = 0
    result = LiveBackendsPhaseExecutor._parse_ferebus_postprocess(
        executor,
        state,
        "FEREBUS",
        [],
    )

    assert result.failure_reason is None
    assert result.state_updates["models_version"] == 0
    committed = campaign / "TRAINED_MODELS" / "iteration-000000"
    assert committed.is_dir()
    for task in manifest["tasks"]:
        model = Model(
            committed
            / str(task["property"])
            / str(task["atom"])
            / Path(str(task["expected_model_path"])).name
        )
        expected_x, expected_y = baseline_by_atom[str(task["atom"])]
        assert np.allclose(np.asarray(model.x, dtype=float), expected_x)
        assert np.allclose(np.asarray(model.y, dtype=float).reshape(-1), expected_y)


def test_stage_ferebus_inputs_clears_stale_models(tmp_path, monkeypatch):
    # A14/A15: a stale .model / *_train.csv from a previous iteration in the shared staging dir must
    # be wiped before this run, not globbed back in.
    campaign = tmp_path / "c"
    cfg = CampaignConfig()
    cfg.system_name = "WATER"
    _prepare_bootstrap_training(campaign, cfg)
    staging = campaign / "TRAINED_MODELS" / "iteration-staging"
    staging.mkdir(parents=True)
    (staging / "STALE.model").write_text("old model from a previous iteration", encoding="utf-8")
    (staging / "Zz9_train.csv").write_text("f1,iqa\n0.1,-1.0\n", encoding="utf-8")
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    stg.stage_ferebus_inputs(
        campaign,
        cfg,
        reference_data_version=0,
        is_initial=False,
    )

    assert not (staging / "STALE.model").exists()
    assert not (staging / "Zz9_train.csv").exists()


def test_stage_ferebus_inputs_rejects_unmanifested_committed_pointdir(tmp_path, monkeypatch):
    campaign = tmp_path / "c"
    cfg = CampaignConfig()
    cfg.system_name = "WATER"
    _prepare_bootstrap_training(campaign, cfg)
    training_dir = campaign / "QM_REFERENCE_DATA" / "iteration-000000"
    rogue = training_dir / "POINT_9999.pointdir"
    rogue.mkdir(parents=True)
    (rogue / "input.gjf").write_text("%chk=x\n", encoding="utf-8")
    monkeypatch.setattr(core_files, "PointsDirectory", _FakePointsDirectory)

    try:
        stg.stage_ferebus_inputs(
            campaign,
            cfg,
            reference_data_version=0,
            is_initial=False,
        )
    except ManifestMismatchError:
        pass
    else:
        raise AssertionError("unmanifested committed pointdir was not rejected")


def test_real_points_directory_exports_iqa_csv_from_aimall_fixtures(tmp_path):
    """Regression for the live INITIAL_FEREBUS handoff.

    Accepted AIMAll pointdirs must expose per-atom properties through
    PointsDirectory.features_with_properties_to_csv(), because FEREBUS staging
    builds its training CSVs from this API.
    """
    from ichor.core.calculators.alf import calculate_alf_atom_sequence
    from ichor.core.files import PointsDirectory

    points = PointsDirectory(FIXTURES / "initial_quantum")
    point_count = sum(1 for _ in points)
    system_alf = points.alf_dict(calculate_alf_atom_sequence)

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        points.features_with_properties_to_csv(
            system_alf,
            str_to_append_to_fname="_train.csv",
            property_types=["iqa"],
        )
    finally:
        os.chdir(cwd)

    csv_paths = sorted(tmp_path.glob("*_train.csv"))
    assert csv_paths
    for csv_path in csv_paths:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert "iqa" in (reader.fieldnames or [])
        assert len(rows) == point_count
        for row in rows:
            assert math.isfinite(float(row["iqa"]))
