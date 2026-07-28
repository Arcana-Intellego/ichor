"""Offline full-cycle tests for the live executor's production wiring."""
import json
import re
from pathlib import Path

import pytest

from ichor.hpc.active_learning.daemon.phase_executor import SBATCH_PHASES
from ichor.hpc.active_learning.daemon.state import CampaignPhase


def test_live_postprocess_refuses_a_deliberately_unsupported_sbatch_phase(
    tmp_path,
    monkeypatch,
):
    """The refusal branch must execute even while all real phases are covered."""
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon import live_executor as live_module
    from types import SimpleNamespace

    cfg = CampaignConfig()
    ex = live_module.LiveBackendsPhaseExecutor(
        campaign_dir=tmp_path / "campaign",
        config=cfg,
        backend_check=False,
    )
    state = SimpleNamespace(iteration=0, campaign_uid="live-smoke")
    unsupported = "TEST_UNSUPPORTED_ARRAY"
    monkeypatch.setattr(
        live_module,
        "SBATCH_PHASES",
        frozenset(set(live_module.SBATCH_PHASES) | {unsupported}),
    )

    with pytest.raises(NotImplementedError, match=unsupported):
        ex.postprocess(state, unsupported, observations=[])


# --- success-path skeleton (skipped until M16) ---




# ==================================================================
# M16 Day 4 P11: end-to-end live executor smoke test
# ==================================================================


def _copy_tree(src_dir, dst_dir):
    """Recursive copy used by _seed_fixture_for_phase. Avoid shutil.copytree
    here so the destination can already exist and we just overlay."""
    from pathlib import Path
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for child in src_dir.iterdir():
        if child.is_file():
            (dst_dir / child.name).write_bytes(child.read_bytes())
        elif child.is_dir():
            _copy_tree(child, dst_dir / child.name)


def _ensure_live_quantum_contract_names(staging_dir):
    """Older fixtures use molecule-prefixed names; live staging expects input.*."""
    from pathlib import Path

    root = Path(staging_dir)
    for pointdir in root.glob("POINT_*.pointdir"):
        for suffix, canonical in (
            ("*.wfn", "input.wfn"),
            ("*.gaussianoutput", "input.gau"),
            ("*.gjf", "input.gjf"),
        ):
            target = pointdir / canonical
            matches = sorted(pointdir.glob(suffix))
            noncanonical_matches = [old for old in matches if old.name != canonical]
            if noncanonical_matches:
                # The daemon-owned GJF is the authoritative input contract.
                # The fixture supplies backend outputs, so it must not replace
                # that GJF with the historical absolute WFN destination
                # embedded in the captured input file.
                if canonical != "input.gjf" or not target.exists():
                    target.write_bytes(noncanonical_matches[0].read_bytes())
            elif matches and not target.exists():
                target.write_bytes(matches[0].read_bytes())
            for old in matches:
                if old.name != canonical:
                    old.unlink()

    # The captured fixture pack predates the minimum valid 2/2/2 FEREBUS
    # bootstrap and contains backend outputs for only three pointdirs.  Clone
    # those backend-owned outputs into any additional daemon-staged pointdirs;
    # retain each point's authoritative GJF and provenance, then retarget the
    # copied WFN/Gaussian geometry in the next helper.
    templates = [
        pointdir
        for pointdir in sorted(root.glob("POINT_*.pointdir"))
        if (pointdir / "input.wfn").is_file()
        and (pointdir / "input.gau").is_file()
    ]
    if templates:
        template = templates[0]
        for pointdir in sorted(root.glob("POINT_*.pointdir")):
            if (pointdir / "input.wfn").is_file() and (
                pointdir / "input.gau"
            ).is_file():
                continue
            for source in template.iterdir():
                destination = pointdir / source.name
                if source.is_dir():
                    if not destination.exists():
                        _copy_tree(source, destination)
                    continue
                if not source.is_file() or source.name in {
                    "input.gjf",
                    ".provenance.json",
                }:
                    continue
                if not destination.exists():
                    destination.write_bytes(source.read_bytes())


def _retarget_live_quantum_fixture_geometry(staging_dir):
    """Make captured backend outputs describe the daemon-staged GJF geometry."""
    from ichor.core.files.gaussian.gjf import GJF
    from ichor.core.files.gaussian.wfn import WFN

    root = Path(staging_dir)
    orientation_row = re.compile(
        r"^(?P<indent>\s*)(?P<centre>\d+)\s+(?P<atomic>\d+)\s+"
        r"(?P<kind>\d+)\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+"
        r"[-+0-9.eEdD]+\s*$"
    )
    for pointdir in root.glob("POINT_*.pointdir"):
        gjf_path = pointdir / "input.gjf"
        wfn_path = pointdir / "input.wfn"
        output_path = pointdir / "input.gau"
        if not (gjf_path.is_file() and wfn_path.is_file() and output_path.is_file()):
            continue

        gjf = GJF(gjf_path)
        target_atoms = list(gjf.atoms)

        # Preserve all captured orbital data while making the WFN nuclei
        # correspond to the input that this fake backend received.
        wfn = WFN(wfn_path, method=str(gjf.method))
        _ = wfn.total_energy
        wfn.atoms = gjf.atoms.to_bohr()
        wfn.write()

        lines = output_path.read_text(encoding="utf-8").splitlines()
        rewritten = []
        for line in lines:
            match = orientation_row.match(line)
            if match is None:
                rewritten.append(line)
                continue
            centre = int(match.group("centre"))
            if centre < 1 or centre > len(target_atoms):
                rewritten.append(line)
                continue
            atom = target_atoms[centre - 1]
            rewritten.append(
                f"{centre:7d}{int(match.group('atomic')):11d}"
                f"{int(match.group('kind')):12d}"
                f"{float(atom.x):16.6f}{float(atom.y):12.6f}"
                f"{float(atom.z):12.6f}"
            )
        output_path.write_text(
            "\n".join(rewritten) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def _prune_unlisted_pointdirs(staging_dir):
    """Keep fixture staging aligned with the submitted POINTS.txt array."""
    from pathlib import Path
    import shutil

    root = Path(staging_dir)
    points = root / "POINTS.txt"
    if not points.is_file():
        return
    allowed = {Path(line.strip()).name for line in points.read_text(encoding="utf-8").splitlines() if line.strip()}
    for pointdir in root.glob("POINT_*.pointdir"):
        if pointdir.name not in allowed:
            shutil.rmtree(pointdir)


def _live_smoke_fixtures():
    """Return the absolute path to the fixture pack."""
    from pathlib import Path
    return Path(__file__).resolve().parent / "fixtures" / "live_outputs"


def _write_loadable_ferebus_model(path, *, atom, alf, ntrain=5, nfeats=3):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.ferebus_prior import (
        resolve_ferebus_prior_contract,
    )

    prior_mean = resolve_ferebus_prior_contract(
        CampaignConfig()
    ).expected_mean_ha("iqa", atom)
    rows = [
        [0.1 + i * 0.1 + j * 0.01 for j in range(nfeats)]
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
        "type constant",
        "value " + repr(prior_mean),
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
    lines += [str(-75.0 - i * 0.01) for i in range(ntrain)]
    lines += ["", "[weights]"]
    lines += ["0.0" for _ in range(ntrain)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_pyferebus_manifest_staging(campaign_dir, reference_data_version=0):
    """Seed the pyferebus-owned staging layout expected by live validators."""
    import json
    import hashlib
    import shutil
    from pathlib import Path

    from ichor.hpc.active_learning.daemon import input_staging as stg
    from ichor.hpc.active_learning.versioning.reference_data import (
        ReferenceDataVersioning,
    )
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.ferebus_prior import (
        backend_kernel_token,
        resolve_ferebus_prior_contract,
        validate_ferebus_config_contract,
    )
    from ichor.hpc.active_learning.versioning.reference_data import (
        canonical_json_sha256,
    )

    prior = resolve_ferebus_prior_contract(CampaignConfig())

    reference_view = ReferenceDataVersioning(
        Path(campaign_dir) / "QM_REFERENCE_DATA"
    ).resolve(int(reference_data_version), verification="deep")
    row_ids = {
        split: [
            index
            for index, entry in enumerate(reference_view.entries)
            if entry.split == split
        ]
        for split in ("train", "int_val", "ext_val")
    }
    row_counts = {split: len(values) for split, values in row_ids.items()}
    source_rows = [
        {"pointdir_name": entry.pointdir_name}
        for entry in reference_view.entries
    ]
    split_rows = {
        split: [source_rows[index] for index in indexes]
        for split, indexes in row_ids.items()
    }
    row_identity_payload = {
        "schema_version": stg.FEREBUS_ROW_IDENTITIES_SCHEMA_VERSION,
        "campaign_uid": reference_view.campaign_uid,
        "reference_data_version": int(reference_data_version),
        "reference_data_view_sha256": reference_view.cumulative_view_sha256,
        "source_rows": source_rows,
        "source_rows_sha256": canonical_json_sha256(source_rows),
        "splits": {
            split: {
                "rows": rows,
                "n_rows": len(rows),
                "row_identity_sha256": canonical_json_sha256(rows),
            }
            for split, rows in split_rows.items()
        },
    }

    target = Path(campaign_dir) / "TRAINED_MODELS" / "iteration-staging"
    if target.exists():
        shutil.rmtree(target)
    specs = {
        "O1": (1, 2, 3),
        "H2": (2, 1, 3),
        "H3": (3, 1, 2),
    }
    tasks = []
    for idx, (atom, alf) in enumerate(specs.items(), start=1):
        model_dir = target / "iqa" / atom
        datasets_dir = model_dir / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        config_path = model_dir / "ferebus.config"
        config_path.write_text(
            "name = \"WATER\"\nproperties = [\"iqa\"]\nmean_type = 21\n"
            + 'level_of_theory = "' + prior.level_of_theory + '"\n'
            + "iqaDeviationFactor = 1.0\nscaling = 1\n"
            + "scale_feats = 1\nscale_prop = 0\n",
            encoding="utf-8",
        )
        parsed_config = validate_ferebus_config_contract(config_path, prior)
        model_path = model_dir / ("WATER_iqa_" + atom + ".model")
        _write_loadable_ferebus_model(
            model_path,
            atom=atom,
            alf=alf,
            ntrain=row_counts["train"],
        )
        train_csv = datasets_dir / ("WATER_" + atom + "_TRAINING_SET.csv")
        int_csv = datasets_dir / ("WATER_" + atom + "_INT_VALIDATION_SET.csv")
        ext_csv = datasets_dir / ("WATER_" + atom + "_EXT_VALIDATION_SET.csv")
        _write_ferebus_metric_csv(train_csv, row_counts["train"])
        _write_ferebus_metric_csv(int_csv, row_counts["int_val"])
        _write_ferebus_metric_csv(ext_csv, row_counts["ext_val"])
        tasks.append({
            "task_index": idx,
            "property": "iqa",
            "atom": atom,
            "prior_mean": prior.task_payload(
                "iqa",
                atom,
                training_values=[0.0] * row_counts["train"],
                training_dataset_sha256=hashlib.sha256(
                    train_csv.read_bytes()
                ).hexdigest(),
            ),
            "alf_1_indexed": list(alf),
            "alf_cli": "_".join(str(value) for value in alf),
            "property_dir": "iqa",
            "output_dir": "iqa/" + atom,
            "input_dir": "iqa/" + atom + "/datasets",
            "config_path": "iqa/" + atom + "/ferebus.config",
            "expected_model_path": "iqa/" + atom + "/WATER_iqa_" + atom + ".model",
            "training_csv": "iqa/" + atom + "/datasets/WATER_" + atom + "_TRAINING_SET.csv",
            "int_validation_csv": "iqa/" + atom + "/datasets/WATER_" + atom + "_INT_VALIDATION_SET.csv",
            "ext_validation_csv": "iqa/" + atom + "/datasets/WATER_" + atom + "_EXT_VALIDATION_SET.csv",
            "command_args": [
                "-c", "iqa/" + atom + "/ferebus.config",
                "-I", "iqa/" + atom + "/datasets",
                "-O", "iqa/" + atom,
                "-P", "iqa",
                "-A", atom,
                "-ALF", "_".join(str(value) for value in alf),
            ],
            "row_counts": dict(row_counts),
            "row_ids": dict(row_ids),
            "datasets": {
                split: {
                    "path": path.relative_to(target).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "rows": row_counts[split],
                    "row_identity_sha256": row_identity_payload["splits"][split][
                        "row_identity_sha256"
                    ],
                    "row_identity_count": row_counts[split],
                }
                for split, path in (
                    ("train", train_csv),
                    ("int_val", int_csv),
                    ("ext_val", ext_csv),
                )
            },
            "generated_config": {
                "path": "iqa/" + atom + "/ferebus.config",
                "size": config_path.stat().st_size,
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "parsed_contract": parsed_config,
                "prior_mean_contract_sha256": prior.contract_sha256,
            },
        })
    row_identity_path = target / stg.FEREBUS_ROW_IDENTITIES
    row_identity_path.write_text(
        json.dumps(row_identity_payload),
        encoding="utf-8",
    )
    split_payload = {
        "schema_version": 7,
        "assignments": {
            reference_view.entries[index].pointdir_name: {"split": split}
            for split, indexes in row_ids.items()
            for index in indexes
        },
    }
    split_path = target / stg.FEREBUS_SPLIT_SNAPSHOT
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    (target / stg.FEREBUS_TASK_MANIFEST).write_text(
        json.dumps({
            "schema_version": stg.FEREBUS_TASK_SCHEMA_VERSION,
            "campaign_uid": reference_view.campaign_uid,
            "system": "WATER",
            "reference_data_version": int(reference_data_version),
            "reference_data_head_manifest_sha256": (
                reference_view.head_manifest_sha256
            ),
            "reference_data_view_sha256": (
                reference_view.cumulative_view_sha256
            ),
            "n_reference_points": len(reference_view.entries),
            "pointdir_row_order": [
                entry.pointdir_name for entry in reference_view.entries
            ],
            "properties": ["iqa"],
            "atoms": list(specs),
            "n_atoms": len(specs),
            "n_tasks": len(tasks),
            "prior_mean_contract": prior.to_dict(),
            "kernel_contract": {
                "family": "rbf",
                "backend_token": backend_kernel_token("rbf"),
                "loss": "huber",
                "constant_noise": True,
                "full_ard": True,
                "feature_scaling": True,
                "property_scaling": False,
                "kernel_prefactor_mode": 2,
            },
            "row_identity_snapshot": {
                "path": stg.FEREBUS_ROW_IDENTITIES,
                "size": row_identity_path.stat().st_size,
                "sha256": hashlib.sha256(row_identity_path.read_bytes()).hexdigest(),
                "source_rows_sha256": row_identity_payload[
                    "source_rows_sha256"
                ],
            },
            "split_ledger": {
                "path": stg.FEREBUS_SPLIT_SNAPSHOT,
                "size": split_path.stat().st_size,
                "sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            },
            "tasks": tasks,
        }),
        encoding="utf-8",
    )
    for name in (stg.FEREBUS_JOB_DETAILS, "commands", "list.txt", "runFerebus.sh"):
        (target / name).write_text(name + "\n", encoding="utf-8")
    from ichor.hpc.active_learning.daemon.ferebus_task_runner import (
        write_preexisting_model_receipts,
    )
    from ichor.hpc.active_learning.submit.pyferebus_wrap import (
        _write_structured_task_map,
    )

    task_map_path = _write_structured_task_map(
        target,
        executable="ferebus",
        execution_kind="synthetic_dry_run",
    )
    task_map = json.loads(task_map_path.read_text(encoding="utf-8"))
    for task in task_map["tasks"]:
        performance_path = target.joinpath(
            *str(task["expected_performance_path"]).split("/")
        )
        performance_path.write_text(
            "RMSE 0.0\nMAE 0.0\ncovariance_condition_number 1.0\n",
            encoding="utf-8",
        )
    write_preexisting_model_receipts(
        target,
        execution_kind="synthetic_dry_run",
    )
    return target


def _write_ferebus_metric_csv(path, n_rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("f1,f2,f3,iqa\n")
        for i in range(int(n_rows)):
            features = [0.1 + i * 0.1 + j * 0.01 for j in range(3)]
            f.write(
                ",".join(str(value) for value in features)
                + ","
                + str(-75.0 - i * 0.01)
                + "\n"
            )


def _ensure_live_trajectory_pool(campaign_dir):
    """Import a tiny pool so live seed selection can publish its manifest."""
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool

    fixture = _live_smoke_fixtures() / "polus_phase_a" / "initial-SAMPLE-2.xyz"
    source = campaign_dir / "live-smoke-pool.xyz"
    fixture_text = fixture.read_text(encoding="utf-8")
    source.write_text(fixture_text * 5, encoding="utf-8", newline="\n")
    TrajectoryPool.import_from(
        source,
        campaign_dir,
        overwrite=True,
    )


def _live_smoke_config(campaign_dir):
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.custom_bootstrap import (
        commit_bootstrap_plan,
        inspect_bootstrap_inputs,
    )

    config = CampaignConfig(max_iterations=1, poll_interval_seconds=1)
    config.point_allocation.bootstrap_training_size = 2
    config.point_allocation.bootstrap_internal_validation_size = 2
    config.point_allocation.bootstrap_external_validation_size = 2
    config.point_allocation.batch_training_size = 1
    config.point_allocation.batch_internal_validation_size = 1
    config.seed_selection.n_seeds_per_iteration = 2
    config.phase_b.descriptor = "rmsd_massweight"
    _ensure_live_trajectory_pool(campaign_dir)
    pool = TrajectoryPool.load(campaign_dir)
    commit_bootstrap_plan(
        inspect_bootstrap_inputs(
            campaign_dir,
            config,
            pool.to_atoms_list(),
            pool_sha256=str(pool.sha256),
        )
    )
    return config


def _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign_dir, call_log):
    """Patch only the expensive pyferebus/FEREBUS submit seam for fixture tests."""
    from pathlib import Path

    from ichor.hpc.active_learning.daemon import input_staging as stg
    from ichor.hpc.active_learning.submit import pyferebus_wrap
    from ichor.hpc.active_learning.submit.pyferebus_wrap import FerebusSubmission

    calls = {"n": 0}

    def fake_stage(campaign_dir_arg, config, reference_data_version):
        is_initial = calls["n"] == 0
        staging = _seed_pyferebus_manifest_staging(
            campaign_dir_arg,
            reference_data_version=0 if is_initial else int(reference_data_version),
        )
        manifest = json.loads(
            (staging / stg.FEREBUS_TASK_MANIFEST).read_text(encoding="utf-8")
        )
        return staging, int(manifest["n_tasks"])

    def fake_submit(jd_file, working_directory, **kwargs):
        working = Path(working_directory)
        script = working / "runFerebus.sh"
        script.write_text("#!/bin/sh\n# fixture pyferebus submit\n", encoding="utf-8")
        phase_name = "INITIAL_FEREBUS" if calls["n"] == 0 else "FEREBUS"
        phase_iteration = 0 if phase_name == "INITIAL_FEREBUS" else 1
        call_log.append((phase_name, phase_iteration))
        calls["n"] += 1
        assert kwargs["overwrite_workdir"] is False
        assert kwargs["move_dataset_files"] is True
        assert (
            "-" + phase_name + "-" + str(phase_iteration) + "-"
            in str(kwargs["expected_job_name"])
        )
        assert int(kwargs["expected_tasks"]) >= 1
        generated = []
        manifest = json.loads(
            (working / stg.FEREBUS_TASK_MANIFEST).read_text(encoding="utf-8")
        )
        generated = [dict(task["generated_config"]) for task in manifest["tasks"]]
        overrides = kwargs["prepared_callback"](working, script, generated)
        submitted_script = Path(overrides["submission_script_path"])
        submitted_script.write_text(
            "#!/bin/sh\n# fixture pyferebus submitted script\n",
            encoding="utf-8",
        )
        from ichor.hpc.active_learning.daemon.script_bundles import (
            AttemptBundle,
            write_script_binding,
        )

        script_binding = write_script_binding(
            AttemptBundle(
                root=submitted_script.parent,
                script=submitted_script,
                outputs=submitted_script.parent / "OUTPUTS",
                errors=submitted_script.parent / "ERRORS",
            )
        )
        kwargs["pre_submit_hook"](submitted_script, script_binding)
        return FerebusSubmission(
            job_id=str(19000 + calls["n"]),
            cluster=None,
            submission_script=submitted_script,
            working_dir=working,
            transfer_learning=False,
            generated_configs=tuple(generated),
            script_binding=script_binding,
        )

    monkeypatch.setattr(stg, "stage_ferebus_inputs", fake_stage)
    monkeypatch.setattr(pyferebus_wrap, "submit_ferebus", fake_submit)


def _annotate_ariadne_fixture_results(campaign_dir, iteration):
    import json as _json
    from ichor.hpc.active_learning.ariadne_outputs import (
        write_optimisation_trajectory,
        write_seed_output_manifest,
    )
    from ichor.hpc.active_learning.acquisition.trajectory_pool import TrajectoryPool
    from ichor.hpc.active_learning.daemon.state import atomic_write_json
    from ichor.hpc.active_learning.layout import active_iteration_dir, ariadne_seed_dir
    from ichor.hpc.active_learning.sampling_protocol import (
        sampling_protocol_audit_path,
        sampling_protocol_resolved_path,
    )
    from ichor.hpc.active_learning.sampling_scale_model import (
        sampling_scale_model_path,
    )
    from ichor.hpc.active_learning.seed_identity import read_ariadne_task_map
    from ichor.hpc.active_learning.versioning.manifest import sha256_file

    iter_dir = active_iteration_dir(campaign_dir, int(iteration))
    task_map = read_ariadne_task_map(iter_dir, expected_iteration=int(iteration))
    resolved_path = sampling_protocol_resolved_path(iter_dir)
    scale_path = sampling_scale_model_path(iter_dir)
    audit_path = sampling_protocol_audit_path(iter_dir)
    resolved_payload = _json.loads(resolved_path.read_text(encoding="utf-8"))
    trajectory_pool = TrajectoryPool.load(campaign_dir)

    def protocol_path(path):
        return path.resolve().relative_to(campaign_dir.resolve()).as_posix()

    fixture_root = _live_smoke_fixtures() / "ariadne_pool"
    for task in task_map["tasks"]:
        array_task_id = int(task["array_task_id"])
        seed_id = int(task["seed_id"])
        source = fixture_root / ("seed_" + str(array_task_id).zfill(4)) / "result.json"
        seed_dir = ariadne_seed_dir(iter_dir, seed_id)
        seed_dir.mkdir(parents=True, exist_ok=False)
        data = _json.loads(source.read_text(encoding="utf-8"))
        coordinates = [list(row) for row in data["final_coordinates"]]
        coordinates[1][0] += 0.40 * (array_task_id + 1)
        coordinates[2][1] += 0.30 * (array_task_id + 1)
        data["final_coordinates"] = coordinates
        data["iteration"] = int(iteration)
        data["seed_id"] = seed_id
        data["seed_uid"] = str(task["seed_uid"])
        data["array_task_id"] = array_task_id
        data["seed_frame_id"] = int(task["frame_id"])
        data["trajectory_sha256"] = str(task_map["trajectory_sha256"])
        seed_atoms = trajectory_pool.frame(int(task["frame_id"]))
        initial_coordinates = [
            [float(atom.x), float(atom.y), float(atom.z)]
            for atom in seed_atoms
        ]
        data["initial_coordinates"] = [list(row) for row in initial_coordinates]
        data["seed_coordinates"] = [list(row) for row in initial_coordinates]
        data["task_success"] = True
        data["sampling_protocol"] = {
            "sampling_aggressiveness": int(
                resolved_payload["sampling_aggressiveness"]
            ),
            "resolved_manifest": protocol_path(resolved_path),
            "resolved_manifest_sha256": sha256_file(resolved_path),
            "scale_model_manifest": protocol_path(scale_path),
            "scale_model_manifest_sha256": sha256_file(scale_path),
            "audit_manifest": protocol_path(audit_path),
            "audit_manifest_sha256": sha256_file(audit_path),
        }
        data["landing_safety"] = {
            "accepted": True,
            "policy": "raw_final",
            "selected_origin": "raw_final",
            "reasons": [],
            "record_only_reasons": [],
            "metrics": {
                "max_displacement_ang": 0.40 * (array_task_id + 1),
                "min_pair_distance_ang": 0.90,
            },
        }
        atomic_write_json(seed_dir / "result.json", data)
        write_optimisation_trajectory(
            seed_dir,
            atom_types=data["atom_types"],
            coordinate_frames=[data["final_coordinates"]],
            alpha_values=[data["alpha_final"]],
            gradient_norms=[0.0],
            origins=["raw_final"],
        )
        write_seed_output_manifest(
            seed_dir,
            campaign_uid=str(task_map["campaign_uid"]),
            iteration=int(iteration),
            seed_id=seed_id,
            seed_uid=str(task["seed_uid"]),
            array_task_id=array_task_id,
            task_success=True,
            task_exit_code=0,
        )


def _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration):
    """Pre-seed the canonical staging dir for the given phase + iteration.

    Called by the mocked sbatch_runner so each phase finds its expected
    artefacts at postprocess time. Returns silently for unknown phases.
    """
    from pathlib import Path
    fixtures = _live_smoke_fixtures()
    campaign_dir = Path(campaign_dir)

    if phase_name in ("PHASE_A_DIVERSITY", "PHASE_B_DIVERSITY"):
        from ichor.hpc.active_learning.sampling.diversity import main as polus_main

        rc = polus_main([
            "--descriptor",
            "rmsd_massweight",
            "--iteration",
            "0" if phase_name == "PHASE_A_DIVERSITY" else str(int(iteration)),
            "--campaign-dir",
            str(campaign_dir),
        ])
        if rc != 0:
            raise RuntimeError(
                "POLUS fixture generation failed for "
                + phase_name
                + " with exit "
                + str(rc)
            )
        return

    if phase_name in ("INITIAL_GAUSSIAN", "INITIAL_AIMALL"):
        target = campaign_dir / ".DATA" / "STAGING" / "initial"
        _copy_tree(fixtures / "initial_quantum", target)
        _ensure_live_quantum_contract_names(target)
        _retarget_live_quantum_fixture_geometry(target)
        _prune_unlisted_pointdirs(target)
    elif phase_name in ("GAUSSIAN", "AIMALL"):
        target = (
            campaign_dir / ".DATA" / "STAGING" / ("iter_" + str(int(iteration)))
        )
        _copy_tree(fixtures / "iter_quantum", target)
        _ensure_live_quantum_contract_names(target)
        _retarget_live_quantum_fixture_geometry(target)
        _prune_unlisted_pointdirs(target)
    elif phase_name in ("INITIAL_FEREBUS", "FEREBUS"):
        _seed_pyferebus_manifest_staging(
            campaign_dir,
            reference_data_version=0 if phase_name == "INITIAL_FEREBUS" else 1,
        )
    elif phase_name == "ARIADNE_ARRAY":
        _annotate_ariadne_fixture_results(campaign_dir, iteration)


def _phase_iteration_from_script(script):
    path = Path(script)
    if path.name == "job.sh":
        iteration_token = path.parent.parent.name
        phase_name = path.parent.parent.parent.name
        if iteration_token.startswith("iteration-"):
            iteration_text = iteration_token.removeprefix("iteration-")
            if iteration_text.isdigit():
                return phase_name, int(iteration_text)
    stem = path.stem
    attempt_match = re.fullmatch(
        r"(?P<phase>.+)-(?P<iteration>\d+)-r\d{4}-a\d{4}-[0-9a-f]+",
        stem,
    )
    if attempt_match is not None:
        return (
            attempt_match.group("phase"),
            int(attempt_match.group("iteration")),
        )
    phase_name, separator, iteration_text = stem.rpartition("-")
    if separator and iteration_text.isdigit():
        return phase_name, int(iteration_text)
    return stem, 0


def _build_live_smoke_sbatch_runner(campaign_dir, call_log):
    """Return a callable suitable for LiveBackendsPhaseExecutor.sbatch_runner.

    The returned callable seeds the staging dir for the inferred phase
    before returning a fake CompletedProcess with a synthetic JobID.
    """
    from pathlib import Path
    from types import SimpleNamespace

    def runner(cmd, **kwargs):
        script = Path(cmd[-1])
        phase_name, iteration = _phase_iteration_from_script(script)
        call_log.append((phase_name, iteration))
        _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration)
        n = len(call_log)
        return SimpleNamespace(
            stdout=str(10000 + n) + "\n",
            stderr="",
            returncode=0,
        )
    return runner


def _build_live_smoke_poller():
    """Return a fake sacct poller that says every job COMPLETED.

    Drop-in for Daemon.sacct_poller; safe to call for any JobID.
    """
    from ichor.hpc.active_learning.submit.sacct_poll import (
        JobObservation, JobStatus,
    )

    def poller(job_id, **kwargs):
        expected = int(kwargs.get("expected_task_count") or 1)
        if expected <= 1:
            return [JobObservation(
                job_id=str(job_id),
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=0,
            )]
        return [
            JobObservation(
                job_id=str(job_id) + "_" + str(idx),
                status=JobStatus.COMPLETED,
                exit_code=(0, 0),
                elapsed_seconds=0,
            )
            for idx in range(expected)
        ]
    return poller



@pytest.mark.integration
def test_live_one_iter_water_tetramer_after_parsers_land(tmp_path, monkeypatch):
    """End-to-end live executor smoke test.

    After M16 Day 3, all 9 SBATCH phases have real parsers registered in
    LIVE_POSTPROCESS_IMPLEMENTED. This test drives Daemon.run() with a
    LiveBackendsPhaseExecutor whose sbatch_runner is a phase-aware mock
    that pre-seeds the canonical staging dirs from the fixture pack, and
    a stub sacct poller that says every job COMPLETED.

    Asserts:
      * Daemon finishes a 1-iteration campaign with rc == 0.
      * No NotImplementedError raised by any live postprocess.
      * Every mandatory submitted phase fired sbatch.
      * Clean allocations do not submit replacement arrays.
      * Final state.phase is DONE and reference_data_version >= 0.
      * Final state.iteration shows the iteration loop ran at least once.
      * journal contains at least one phase_succeeded_live event.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.live_executor import (
        LIVE_POSTPROCESS_IMPLEMENTED,
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.phase_executor import SBATCH_PHASES
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, read_state,
    )
    from ichor.hpc.active_learning.daemon.journal import iter_events

    # Sanity: all SBATCH phases must be implemented after M16 Day 3.
    assert SBATCH_PHASES.issubset(LIVE_POSTPROCESS_IMPLEMENTED)

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = _live_smoke_config(campaign)
    cfg.to_yaml(campaign / "campaign.yaml")

    call_log = []
    _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign, call_log)
    sbatch_runner = _build_live_smoke_sbatch_runner(campaign, call_log)
    poller = _build_live_smoke_poller()

    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign,
        config=cfg,
        sbatch_runner=sbatch_runner,
        backend_check=False,
    )
    d = Daemon(
        campaign_dir=campaign,
        config=cfg,
        executor=ex,
        sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0, "daemon exited non-zero: " + str(rc)

    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE, (
        "expected DONE, got " + str(state.phase)
    )
    # max_iterations=1 runs active iteration 1 exactly once, then finishes.
    assert int(state.iteration) == 1
    # INITIAL_FEREBUS commits initial models 0 + iteration-1 FEREBUS
    # commits models 1, so both versions should be at least 1.
    assert state.reference_data_version >= 1, (
        "reference_data_version=" + str(state.reference_data_version)
        + "; expected at least 1 (initial + 1 iter)"
    )
    assert state.models_version >= 1, (
        "models_version=" + str(state.models_version)
        + "; expected at least 1 (INITIAL_FEREBUS + FEREBUS)"
    )
    from ichor.hpc.active_learning.versioning.trained_models import (
        TrainedModelVersioning,
    )

    model_versions = TrainedModelVersioning(campaign / "TRAINED_MODELS")
    assert model_versions.list_committed_versions() == [0, 1]
    initial_models = model_versions.resolve(0, verification="deep")
    current_models = model_versions.resolve(1, verification="deep")
    assert current_models.parent_version == 0
    assert current_models.parent_manifest_sha256 == initial_models.head_manifest_sha256
    assert current_models.reference_data_version == state.reference_data_version
    assert current_models.reference_data_view_sha256
    assert model_versions.current_version() == 1
    for model_set in (initial_models, current_models):
        assert not (model_set.root / "task_artefacts").exists()
        assert not list(model_set.root.glob("*.model"))
        assert not list(model_set.root.glob("*.config"))
        assert all(task.model.path.parent == task.config.path.parent for task in model_set.tasks)

    phases_called = {p for p, _ in call_log}
    mandatory_submissions = {
        "PHASE_A_DIVERSITY",
        "INITIAL_GAUSSIAN",
        "INITIAL_FEREBUS",
        "ARIADNE_ARRAY",
        "PHASE_B_DIVERSITY",
        "GAUSSIAN",
        "FEREBUS",
    }
    assert mandatory_submissions.issubset(phases_called), (
        "mandatory sbatch phases missing: "
        + str(sorted(mandatory_submissions - phases_called))
    )
    replacement_phases = {
        "INITIAL_REPLACEMENT_GAUSSIAN",
        "INITIAL_REPLACEMENT_AIMALL",
        "REPLACEMENT_GAUSSIAN",
        "REPLACEMENT_AIMALL",
    }
    assert phases_called.isdisjoint(replacement_phases)

    # Live parsers emitted phase_succeeded_live events.
    journal_path = (
        campaign / ".DATA" / "ACTIVE_LEARNING" / "journal.ndjson"
    )
    events = list(iter_events(journal_path))
    live_succeeded = [
        e for e in events if e.get("event") == "phase_succeeded_live"
    ]
    assert live_succeeded, (
        "no phase_succeeded_live events emitted; live parsers may have "
        "silently fallen through. Saw events: "
        + str(sorted({e.get("event") for e in events}))
    )



def _build_no_wd_sbatch_runner(campaign_dir, call_log):
    """Like the standard live-smoke runner but strips the
    whitened_distance_final key from each ARIADNE seed result.json
    immediately after the standard seeder writes them.

    Used by the fallback-path test to verify the live parser copes when
    the field is missing -- older runs, hand-crafted JSON, or any
    fixture pack predating Phase B.
    """
    import json as _json
    from pathlib import Path
    from types import SimpleNamespace

    def runner(cmd, **kwargs):
        script = Path(cmd[-1])
        phase_name, iteration = _phase_iteration_from_script(script)
        call_log.append((phase_name, iteration))
        _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration)

        # for ARIADNE_ARRAY, strip the whitened_distance_final key from
        # every per-seed result.json so the parser is forced through the
        # synthetic fallback branch.
        if phase_name == "ARIADNE_ARRAY":
            from ichor.hpc.active_learning.ariadne_outputs import (
                SEED_OUTPUT_MANIFEST_FILENAME,
                write_seed_output_manifest,
            )
            from ichor.hpc.active_learning.daemon.state import atomic_write_json
            from ichor.hpc.active_learning.layout import (
                active_iteration_dir,
                ariadne_seeds_dir,
            )

            iter_dir = active_iteration_dir(campaign_dir, iteration)
            for seed_dir in ariadne_seeds_dir(iter_dir).iterdir():
                result_path = seed_dir / "result.json"
                output_path = seed_dir / SEED_OUTPUT_MANIFEST_FILENAME
                try:
                    data = _json.loads(result_path.read_text(encoding="utf-8"))
                    output = _json.loads(output_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                data.pop("whitened_distance_final", None)
                atomic_write_json(result_path, data)
                write_seed_output_manifest(
                    seed_dir,
                    campaign_uid=str(output["campaign_uid"]),
                    iteration=int(output["iteration"]),
                    seed_id=int(output["seed_id"]),
                    seed_uid=str(output["seed_uid"]),
                    array_task_id=int(output["array_task_id"]),
                    task_success=bool(output["task_success"]),
                    task_exit_code=int(output["task_exit_code"]),
                )

        n = len(call_log)
        return SimpleNamespace(
            stdout=str(10000 + n) + chr(10),
            stderr="",
            returncode=0,
        )
    return runner


@pytest.mark.integration
def test_live_one_iter_whitened_distance_fallback(tmp_path, monkeypatch):
    """Same end-to-end flow as the main F14 test, but each ARIADNE
    result.json has whitened_distance_final stripped. The daemon should
    still complete cleanly -- the live parser falls back to the synthetic
    alpha-delta proxy in _synthetic_whitened_distance.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.live_executor import (
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, read_state,
    )

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = _live_smoke_config(campaign)
    cfg.to_yaml(campaign / "campaign.yaml")

    call_log = []
    _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign, call_log)
    runner = _build_no_wd_sbatch_runner(campaign, call_log)
    poller = _build_live_smoke_poller()

    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign, config=cfg,
        sbatch_runner=runner, backend_check=False,
    )
    d = Daemon(
        campaign_dir=campaign, config=cfg,
        executor=ex, sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0, "daemon exited non-zero: " + str(rc)

    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE
    # the daemon still progresses; the fallback path is exercised inside
    # _parse_ariadne_array_postprocess without us needing to inspect the
    # actual whitened-distance values here.


def _build_both_phase_b_sbatch_runner(campaign_dir, call_log):
    """Like the standard live-smoke runner but after PHASE_B_DIVERSITY
    standard seeding, additionally writes a different selected_raw.xyz with
    a deliberately different frame count.

    Used by the dedup-naming test to verify that the parser picks the
    deduplicated selected.xyz rather than the raw one.
    """
    from pathlib import Path
    from types import SimpleNamespace

    def runner(cmd, **kwargs):
        script = Path(cmd[-1])
        phase_name, iteration = _phase_iteration_from_script(script)
        call_log.append((phase_name, iteration))
        _live_smoke_seed_for_phase(campaign_dir, phase_name, iteration)

        if phase_name == "PHASE_B_DIVERSITY":
            from ichor.hpc.active_learning.layout import (
                active_iteration_dir,
                active_phase_b_dir,
            )

            iter_dir = active_iteration_dir(campaign_dir, iteration)
            raw = active_phase_b_dir(iter_dir) / "selected_raw.xyz"
            xyz_lines = []
            for k in range(4):
                xyz_lines.append("3")
                xyz_lines.append("raw frame " + str(k))
                xyz_lines.append("O 0.0 0.0 0.0")
                xyz_lines.append("H 0.96 0.0 0.0")
                xyz_lines.append("H -0.24 0.93 0.0")
            raw.write_text(chr(10).join(xyz_lines) + chr(10), encoding="utf-8")

        n = len(call_log)
        return SimpleNamespace(
            stdout=str(10000 + n) + chr(10),
            stderr="",
            returncode=0,
        )
    return runner


@pytest.mark.integration
def test_live_one_iter_prefers_dedup_filtered_sample(tmp_path, monkeypatch):
    """When both selected.xyz and selected_raw.xyz exist
    (with different frame counts), the live parser should pick the
    deduplicated selected.xyz -- that is the canonical name the daemon
    main always writes; raw is only retained as a debugging artefact.
    """
    from ichor.hpc.active_learning.config import CampaignConfig
    from ichor.hpc.active_learning.daemon.daemon import Daemon
    from ichor.hpc.active_learning.daemon.live_executor import (
        LiveBackendsPhaseExecutor,
    )
    from ichor.hpc.active_learning.daemon.state import (
        CampaignPhase, read_state,
    )
    from ichor.hpc.active_learning.daemon.journal import iter_events

    campaign = tmp_path / "campaign"
    campaign.mkdir()
    cfg = _live_smoke_config(campaign)
    cfg.to_yaml(campaign / "campaign.yaml")

    call_log = []
    _patch_ferebus_submit_for_live_smoke(monkeypatch, campaign, call_log)
    runner = _build_both_phase_b_sbatch_runner(campaign, call_log)
    poller = _build_live_smoke_poller()

    ex = LiveBackendsPhaseExecutor(
        campaign_dir=campaign, config=cfg,
        sbatch_runner=runner, backend_check=False,
    )
    d = Daemon(
        campaign_dir=campaign, config=cfg,
        executor=ex, sacct_poller=poller,
        sleep_fn=lambda s: None,
    )
    rc = d.run(max_ticks=500, catch_keyboard_interrupt=False)
    assert rc == 0
    state = read_state(d.state_path())
    assert state.phase is CampaignPhase.DONE

    # inspect the journal for PHASE_B_DIVERSITY phase_succeeded_live event.
    # its sample_path field should end with selected.xyz, not selected_raw.xyz.
    journal_path = (
        campaign / ".DATA" / "ACTIVE_LEARNING"
        / "journal.ndjson"
    )
    events = list(iter_events(journal_path))
    phase_b_events = [
        e for e in events
        if e.get("event") == "phase_succeeded_live"
        and e.get("phase") == "PHASE_B_DIVERSITY"
    ]
    assert phase_b_events, "no PHASE_B_DIVERSITY phase_succeeded_live event"
    last = phase_b_events[-1]
    sample_path = str(last.get("sample_path"))
    assert sample_path.endswith("selected.xyz"), (
        "expected selected.xyz but got " + sample_path
    )
    # also confirm the fixture-supplied frame count (2) was parsed, not the
    # 4-frame raw file we deliberately also wrote.
    assert int(last.get("n_frames", 0)) == 2
