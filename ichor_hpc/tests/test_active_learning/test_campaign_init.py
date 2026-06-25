from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ichor.hpc.active_learning.cli import cmd_import_pool, cmd_init
from ichor.hpc.active_learning.config import CampaignConfig


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "water_tetramer.xyz"


def _args(campaign, source=None, *, force=False, no_filter=True):
    from argparse import Namespace

    return Namespace(
        campaign_dir=str(campaign) if campaign is not None else None,
        source=str(source) if source is not None else None,
        force=force,
        no_outlier_filter=no_filter,
    )


def test_init_populates_missing_campaign_yaml_and_imports_default_pool(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")

    rc = cmd_init(_args(tmp_path))

    assert rc == 0
    cfg = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert cfg.schema_version == 3
    assert cfg.max_iterations == 1
    assert (tmp_path / ".DATA" / "TRAJECTORY" / "pool.xyz").is_file()


def test_init_fills_sparse_campaign_yaml_preserving_user_override(tmp_path):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")
    (tmp_path / "campaign.yaml").write_text(
        "schema_version: 3\n"
        "system_name: MY_SYSTEM\n"
        "gaussian:\n"
        "  basis_set: def2-SVP\n",
        encoding="utf-8",
    )

    rc = cmd_init(_args(tmp_path))

    assert rc == 0
    cfg = CampaignConfig.from_yaml(tmp_path / "campaign.yaml")
    assert cfg.system_name == "MY_SYSTEM"
    assert cfg.gaussian.basis_set == "def2-SVP"
    assert cfg.seed_selection.n_seeds_per_iteration == 4
    raw = (tmp_path / "campaign.yaml").read_text(encoding="utf-8")
    assert "# campaign.yaml" in raw
    assert "seed_selection:" in raw


def test_init_reports_missing_default_pool(tmp_path, capsys):
    rc = cmd_init(_args(tmp_path))

    assert rc == 2
    assert "Put pool.xyz in the campaign directory" in capsys.readouterr().err


def test_import_pool_deprecated_alias_calls_init(tmp_path, capsys):
    if not FIXTURE.is_file():
        pytest.skip("water_tetramer.xyz fixture missing")
    shutil.copy(FIXTURE, tmp_path / "pool.xyz")

    rc = cmd_import_pool(_args(tmp_path))

    assert rc == 0
    assert "import-pool is deprecated" in capsys.readouterr().err
    assert (tmp_path / "campaign.yaml").is_file()
    assert (tmp_path / ".DATA" / "TRAJECTORY" / "pool.xyz").is_file()
