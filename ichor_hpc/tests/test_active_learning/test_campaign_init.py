from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ichor.hpc.active_learning.cli import cmd_import_pool, cmd_init
from ichor.hpc.active_learning.config import CONFIG_SCHEMA_VERSION, CampaignConfig
from ichor.hpc.active_learning.daemon.daemon import DEFAULT_DATA_SUBDIR
from ichor.hpc.active_learning.daemon.state import (
    DEFAULT_STATE_FILENAME,
    fresh_campaign_state,
    write_state,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"


def _args(campaign, source=None, *, force=False):
    from argparse import Namespace

    return Namespace(
        campaign_dir=str(campaign) if campaign is not None else None,
        source=str(source) if source is not None else None,
        force=force,
    )


def _write_first_fixture_frame(path):
    from ichor.core.files.xyz import Trajectory

    trajectory = Trajectory(FIXTURE)
    trajectory.read()
    frame = next(iter(trajectory))
    lines = [str(len(frame)), "anchor fixture"]
    lines.extend(
        atom.type
        + " "
        + str(float(atom.x))
        + " "
        + str(float(atom.y))
        + " "
        + str(float(atom.z))
        for atom in frame
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def test_init_populates_missing_campaign_yaml_and_imports_default_pool(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")

    rc = cmd_init(_args(tmp_path))

    assert rc == 0
    cfg = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert cfg.schema_version == CONFIG_SCHEMA_VERSION
    assert cfg.max_iterations == 1
    assert (tmp_path / ".DATA" / "TRAJECTORY" / "pool.xyz").is_file()


def test_init_fills_sparse_campaign_yaml_preserving_user_override(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    (tmp_path / "campaign.yaml").write_text(
        "schema_version: 10\n"
        "campaign:\n"
        "  system_name: MY_SYSTEM\n"
        "gaussian:\n"
        "  basis_set: def2-SVP\n",
        encoding="utf-8",
    )

    rc = cmd_init(_args(tmp_path))

    assert rc == 0
    cfg = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert cfg.system_name == "MY_SYSTEM"
    assert cfg.gaussian.basis_set == "def2-SVP"
    assert cfg.seed_selection.n_seeds_per_iteration == 8
    raw = (tmp_path / "campaign.yaml").read_text(encoding="utf-8")
    assert "# campaign.yaml" in raw
    assert "seed_selection:" in raw


def test_init_refuses_infeasible_pool_before_state_creation(tmp_path, capsys):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    (tmp_path / "campaign.yaml").write_text(
        "schema_version: 10\n"
        "campaign:\n"
        "  max_iterations: 2\n"
        "point_allocation:\n"
        "  bootstrap_training_size: 8\n"
        "  bootstrap_internal_validation_size: 2\n"
        "  bootstrap_external_validation_size: 2\n"
        "  batch_training_size: 3\n"
        "  batch_internal_validation_size: 1\n"
        "seed_selection:\n"
        "  n_seeds_per_iteration: 8\n",
        encoding="utf-8",
    )

    rc = cmd_init(_args(tmp_path))

    assert rc == 17
    err = capsys.readouterr().err
    assert "trajectory pool is infeasible" in err
    assert "12 + 2 * 8 = 28" in err
    assert not (
        tmp_path / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    ).exists()


def test_init_reports_missing_default_pool(tmp_path, capsys):
    rc = cmd_init(_args(tmp_path))

    assert rc == 2
    err = capsys.readouterr().err
    assert "campaign.source_path" in err
    assert (tmp_path / "campaign.yaml").is_file()
    assert not (
        tmp_path / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    ).is_file()


def test_init_is_idempotent_after_operator_pool_is_imported(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")

    assert cmd_init(_args(tmp_path)) == 0
    (tmp_path / "pool.xyz").unlink()

    assert cmd_init(_args(tmp_path)) == 0


def test_import_pool_deprecated_alias_calls_init(tmp_path, capsys):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")

    rc = cmd_import_pool(_args(tmp_path))

    assert rc == 0
    assert "import-pool is deprecated" in capsys.readouterr().err
    assert (tmp_path / "campaign.yaml").is_file()
    assert (tmp_path / ".DATA" / "TRAJECTORY" / "pool.xyz").is_file()


def test_force_pool_import_refuses_started_campaign(tmp_path, capsys):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    assert cmd_init(_args(tmp_path)) == 0
    data = tmp_path / DEFAULT_DATA_SUBDIR
    data.mkdir(parents=True, exist_ok=True)
    write_state(data / DEFAULT_STATE_FILENAME, fresh_campaign_state())

    rc = cmd_init(_args(tmp_path, source=FIXTURE, force=True))

    err = capsys.readouterr().err
    assert rc == 16
    assert "refusing to overwrite the trajectory pool" in err
    assert "state.json" in err


def test_init_imports_configured_anchor_into_campaign_owned_storage(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    _write_first_fixture_frame(tmp_path / "operator-anchor.xyz")
    cfg = CampaignConfig()
    cfg.point_allocation.anchor = True
    cfg.campaign.anchor_path = "operator-anchor.xyz"
    cfg.to_yaml(tmp_path / "campaign.yaml")

    rc = cmd_init(_args(tmp_path))

    assert rc == 0
    canonical = tmp_path / ".DATA" / "TRAJECTORY" / "anchor.xyz"
    manifest = tmp_path / ".DATA" / "TRAJECTORY" / "ANCHOR_SOURCE.json"
    assert canonical.is_file()
    assert manifest.is_file()


def test_init_anchor_import_is_idempotent_after_operator_source_is_removed(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    source = tmp_path / "operator-anchor.xyz"
    _write_first_fixture_frame(source)
    cfg = CampaignConfig()
    cfg.point_allocation.anchor = True
    cfg.campaign.anchor_path = source.name
    cfg.to_yaml(tmp_path / "campaign.yaml")

    assert cmd_init(_args(tmp_path)) == 0
    source.unlink()

    assert cmd_init(_args(tmp_path)) == 0


def test_init_fails_when_anchor_is_enabled_but_configured_path_is_missing(
    tmp_path,
    capsys,
):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    cfg = CampaignConfig()
    cfg.point_allocation.anchor = True
    cfg.campaign.anchor_path = "missing-anchor.xyz"
    cfg.to_yaml(tmp_path / "campaign.yaml")

    rc = cmd_init(_args(tmp_path))

    assert rc == 18
    assert "campaign.anchor_path" in capsys.readouterr().err
    assert not (
        tmp_path / ".DATA" / "ACTIVE_LEARNING" / DEFAULT_STATE_FILENAME
    ).is_file()


def test_init_resolves_environment_variable_in_campaign_source_path(
    tmp_path,
    monkeypatch,
):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    external = tmp_path / "external-pool.xyz"
    shutil.copy(FIXTURE, external)
    monkeypatch.setenv("ICHOR_TEST_POOL", str(external))
    cfg = CampaignConfig()
    cfg.campaign.source_path = "$ICHOR_TEST_POOL"
    cfg.to_yaml(tmp_path / "campaign.yaml")

    assert cmd_init(_args(tmp_path)) == 0
    assert (tmp_path / ".DATA" / "TRAJECTORY" / "pool.xyz").is_file()
