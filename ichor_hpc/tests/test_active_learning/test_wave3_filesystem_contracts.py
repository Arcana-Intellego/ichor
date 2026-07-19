from __future__ import annotations

import ast
import errno
import inspect
import os
from pathlib import Path

import pytest

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import Daemon
from ichor.hpc.active_learning.daemon.dry_run_executor import DryRunPhaseExecutor
from ichor.hpc.active_learning.daemon.filesystem import (
    campaign_owned_path,
    operational_data_dir,
    operational_path,
)
from ichor.hpc.active_learning.daemon.resource_records import (
    resolution_payload,
    write_resolution,
)
from ichor.hpc.active_learning.daemon.resource_solver import ResolvedPhaseResources
from ichor.hpc.active_learning.daemon.scratch import prepare_task_scratch
from ichor.hpc.active_learning.daemon.script_bundles import (
    prepare_attempt_bundle,
    write_attempt_script,
    write_script_binding,
)
from ichor.hpc.active_learning.daemon.state import (
    atomic_write_text,
    fresh_campaign_state,
    read_state,
    write_state,
)
from ichor.hpc.active_learning.daemon.submission_intent import intent_dir
from ichor.hpc.active_learning.layout import (
    parse_staging_pointdir_name,
    staging_pointdir_name,
)
from ichor.hpc.active_learning.strict_json import loads
from ichor.hpc.active_learning.operator_paths import (
    reject_operator_input_symlinks,
    resolve_campaign_input_path,
)
from ichor.hpc.active_learning.versioning.manifest import fsync_regular_files
from ichor.hpc.active_learning.versioning.versioned_directory import VersionedDirectory


def test_staging_point_names_use_minimum_width_and_round_trip():
    from ichor.hpc.active_learning.daemon.input_staging import (
        _validate_pointdir_basename,
    )

    expected = {
        9999: "POINT_9999.pointdir",
        10000: "POINT_10000.pointdir",
        25000: "POINT_25000.pointdir",
        1000000: "POINT_1000000.pointdir",
    }
    for value, name in expected.items():
        assert staging_pointdir_name(value) == name
        assert parse_staging_pointdir_name(name) == value
        assert _validate_pointdir_basename(name) == name
    with pytest.raises(ValueError, match="noncanonical"):
        parse_staging_pointdir_name("POINT_00001.pointdir")


def test_executor_does_not_expose_an_ignored_active_directory_override():
    assert "al_dir_name" not in inspect.signature(DryRunPhaseExecutor).parameters


def test_strict_json_rejects_top_level_nested_duplicates_and_non_finite():
    with pytest.raises(ValueError, match="duplicate object key 'phase'"):
        loads('{"phase":"HALTED","phase":"INIT"}')
    with pytest.raises(ValueError, match="duplicate object key 'accepted'"):
        loads('{"result":{"accepted":false,"accepted":true}}')
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        loads('{"value":NaN}')


def test_state_reader_rejects_duplicate_authoritative_keys(tmp_path):
    path = tmp_path / "state.json"
    write_state(path, fresh_campaign_state(campaign_uid="uid"))
    text = path.read_text(encoding="utf-8")
    text = text.replace('"phase": "INIT"', '"phase": "HALTED",\n  "phase": "INIT"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object key 'phase'"):
        read_state(path)


def test_active_learning_sources_do_not_import_permissive_stdlib_json():
    root = (
        Path(__file__).resolve().parents[2]
        / "ichor"
        / "hpc"
        / "active_learning"
    )
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "strict_json.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "json" for alias in node.names
            ):
                offenders.append(str(path.relative_to(root)))
            if isinstance(node, ast.ImportFrom) and node.module == "json":
                offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_atomic_write_propagates_real_file_fsync_failure(monkeypatch, tmp_path):
    import ichor.hpc.active_learning.daemon.state as state_module

    def fail_fsync(_descriptor):
        raise OSError(errno.EIO, "injected storage failure")

    monkeypatch.setattr(state_module.os, "fsync", fail_fsync)
    target = tmp_path / "state.json"
    with pytest.raises(OSError) as error:
        atomic_write_text(target, "{}\n")
    assert error.value.errno == errno.EIO
    assert not target.exists()


def test_atomic_write_ignores_only_unsupported_fsync(monkeypatch, tmp_path):
    import ichor.hpc.active_learning.daemon.state as state_module

    def unsupported(_descriptor):
        raise OSError(errno.EINVAL, "unsupported")

    monkeypatch.setattr(state_module.os, "fsync", unsupported)
    target = tmp_path / "state.json"
    atomic_write_text(target, "{}\n")
    assert target.read_text(encoding="utf-8") == "{}\n"


def test_manifest_file_fsync_failure_is_not_suppressed(monkeypatch, tmp_path):
    import ichor.hpc.active_learning.daemon.state as state_module

    (tmp_path / "data.bin").write_bytes(b"data")

    def fail_fsync(_descriptor):
        raise OSError(errno.ENOSPC, "injected full filesystem")

    monkeypatch.setattr(state_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError) as error:
        fsync_regular_files(tmp_path)
    assert error.value.errno == errno.ENOSPC


def test_operational_paths_reject_symlinked_data_ancestor(tmp_path):
    campaign = tmp_path / "campaign"
    outside = tmp_path / "outside"
    campaign.mkdir()
    outside.mkdir()
    try:
        os.symlink(outside, campaign / ".DATA", target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(ValueError, match="symlink"):
        Daemon(campaign_dir=campaign, config=CampaignConfig()).data_dir()
    with pytest.raises(ValueError, match="symlink"):
        intent_dir(campaign)


def test_campaign_owned_relative_paths_are_independent_of_working_directory(
    monkeypatch,
    tmp_path,
):
    campaign = (tmp_path / "campaign").resolve()
    nested = campaign / "nested" / "working"
    unrelated = tmp_path / "unrelated"
    nested.mkdir(parents=True)
    unrelated.mkdir()
    expected_data = campaign / ".DATA" / "ACTIVE_LEARNING"

    for working_directory in (campaign, nested, unrelated):
        monkeypatch.chdir(working_directory)
        assert campaign_owned_path(
            campaign,
            Path(".DATA") / "ACTIVE_LEARNING",
        ) == expected_data
        assert operational_data_dir(campaign) == expected_data
        assert operational_path(campaign, "state.json") == expected_data / "state.json"
        assert intent_dir(campaign) == expected_data / "submission_intents"

        daemon = Daemon(campaign_dir=campaign, config=CampaignConfig())
        assert daemon.state_path() == expected_data / "state.json"
        assert daemon.lock_path() == expected_data / "daemon.lock"
        assert daemon.journal_path() == expected_data / "journal.ndjson"


def test_campaign_owned_absolute_paths_and_escape_rejection_are_unchanged(tmp_path):
    campaign = (tmp_path / "campaign").resolve()
    campaign.mkdir()
    owned = campaign / ".DATA" / "ACTIVE_LEARNING"

    assert campaign_owned_path(campaign, owned) == owned
    with pytest.raises(ValueError, match="escapes campaign root"):
        campaign_owned_path(campaign, Path("..") / "outside")
    with pytest.raises(ValueError, match="escapes campaign root"):
        campaign_owned_path(campaign, tmp_path / "outside")


def test_state_reader_and_writer_reject_symlinked_campaign_state(tmp_path):
    campaign = tmp_path / "campaign"
    data = campaign / ".DATA" / "ACTIVE_LEARNING"
    outside = tmp_path / "outside-state.json"
    data.mkdir(parents=True)
    write_state(outside, fresh_campaign_state(campaign_uid="outside"))
    state_path = data / "state.json"
    try:
        os.symlink(outside, state_path)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(ValueError, match="symlink"):
        read_state(state_path)
    with pytest.raises(ValueError, match="symlink"):
        write_state(state_path, fresh_campaign_state(campaign_uid="campaign"))
    assert read_state(outside).campaign_uid == "outside"


def test_operator_input_resolution_preserves_and_rejects_parent_symlink(tmp_path):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    source = real_parent / "pool.xyz"
    source.write_text("1\nframe\nH 0 0 0\n", encoding="utf-8")
    linked_parent = tmp_path / "linked"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    lexical = resolve_campaign_input_path(tmp_path, linked_parent / "pool.xyz")
    assert lexical == linked_parent / "pool.xyz"
    with pytest.raises(ValueError, match="symlink"):
        reject_operator_input_symlinks(lexical)


def test_scratch_permission_failure_halts_before_task_metadata(
    monkeypatch, tmp_path,
):
    identity = "r0000-a0001-deadbeef"
    payload = resolution_payload(
        campaign_uid="uid",
        phase_name="GAUSSIAN",
        iteration=1,
        attempt_id="attempt-id",
        submission_identity=identity,
        resolved=ResolvedPhaseResources(
            backend="gaussian",
            partition="multicore",
            ntasks=1,
            cpus_per_task=1,
            mem_per_cpu="4G",
            estimated_total_memory_gb=4.0,
            partition_memory_per_core_gb=4.0,
            cpus_raw=1,
            mem_per_cpu_raw="4G",
            cpu_reason="fixture",
            memory_reason="fixture",
        ),
        evidence={"source": "fixture"},
        scratch_path_template="fixture",
    )
    binding = write_resolution(tmp_path, payload)
    bundle = prepare_attempt_bundle(
        tmp_path,
        "GAUSSIAN",
        1,
        identity,
        array_size=1,
        max_log_files_per_directory=10,
    )
    write_attempt_script(bundle, "#!/bin/bash\ntrue\n")
    script_binding = write_script_binding(bundle)
    real_chmod = Path.chmod

    def fail_leaf_chmod(path, mode):
        if path.name == "task-0":
            raise OSError("injected chmod failure")
        return real_chmod(path, mode)

    monkeypatch.setattr(Path, "chmod", fail_leaf_chmod)
    with pytest.raises(OSError, match="injected chmod failure"):
        prepare_task_scratch(
            tmp_path,
            campaign_uid="uid",
            phase_name="GAUSSIAN",
            iteration=1,
            attempt_id="attempt-id",
            submission_identity=identity,
            job_id="100",
            array_task_id=0,
            resource_resolution_path=binding["path"],
            resource_resolution_sha256=binding["sha256"],
            script_binding_path=script_binding["path"],
            script_binding_sha256=script_binding["sha256"],
        )
    assert not any(tmp_path.rglob("TASK.json"))


def test_idempotent_commit_propagates_stale_staging_cleanup_failure(
    monkeypatch, tmp_path,
):
    versioning = VersionedDirectory(tmp_path / "versions")
    staging = versioning.stage(None, 0)
    (staging / "data.txt").write_text("committed", encoding="utf-8")
    versioning.commit(0)
    staging = versioning.stage(None, 0)
    (staging / "stale.txt").write_text("stale", encoding="utf-8")

    def fail_cleanup(_path):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(
        "ichor.hpc.active_learning.versioning.versioned_directory.shutil.rmtree",
        fail_cleanup,
    )
    with pytest.raises(OSError, match="injected cleanup failure"):
        versioning.commit(0)
    assert staging.is_dir()
