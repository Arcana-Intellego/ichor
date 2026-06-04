"""Legacy ferebus.toml writer + FEREBUS script-render guard.

These are off-cluster contract checks (format + script rendering). The actual
FEREBUS binary run is cluster-only.
"""
from pathlib import Path

from ichor.hpc.active_learning.config import CampaignConfig
from ichor.hpc.active_learning.daemon.ferebus_config import write_ferebus_toml
from ichor.hpc.active_learning.daemon.live_executor import build_sbatch_script


def test_ferebus_toml_format(tmp_path):
    p = write_ferebus_toml(
        tmp_path / "ferebus.toml",
        system_name="WATER", natoms=3, properties=("iqa",), kernel="rbfc_per",
        atom="O1", alf=(1, 2, 3), training=1, validation=1,
    )
    t = p.read_text(encoding="utf-8")
    assert 'name = "WATER"' in t
    assert "natoms = 3" in t
    assert 'properties = ["iqa"]' in t
    # training/validation are FEREBUS on/off booleans, written 0/1
    assert "training = 1" in t
    assert "validation = 1" in t
    assert 'atoms = [{name="O1", alf=[1, 2, 3]}]' in t
    # LF only -- a stray CR can break the Fortran parser's last-token match
    assert "\r" not in t


def test_ferebus_toml_validation_off(tmp_path):
    p = write_ferebus_toml(
        tmp_path / "ferebus.toml",
        system_name="X",
        natoms=2,
        atom="O1",
        alf=(1, 2, 3),
        validation=0,
    )
    assert "validation = 0" in p.read_text(encoding="utf-8")


def test_ferebus_sbatch_renderer_is_not_the_live_submit_path():
    body = build_sbatch_script(
        phase_name="FEREBUS", iteration=1, campaign_dir=Path("/scratch/camp"),
        config=CampaignConfig(), array_size=None,
    )
    assert "pyferebus wrapper" in body
    assert "ferebus_${ATOM}.toml" not in body
    assert "runFerebus.py" not in body


def test_initial_ferebus_renderer_is_not_the_live_submit_path():
    body = build_sbatch_script(
        phase_name="INITIAL_FEREBUS", iteration=0, campaign_dir=Path("/c"),
        config=CampaignConfig(), array_size=None,
    )
    assert "pyferebus wrapper" in body
    assert "ferebus_${ATOM}.toml" not in body


def test_ferebus_toml_writes_target_prop(tmp_path):
    # when we hand it the per-atom iqa stats they land as target_prop_* lines (FEREBUS leans on
    # them for scaling). without them the keys are simply omitted -- see the format test above.
    stats = {"min": -75.5, "max": -75.0, "range": 0.5,
             "mean": -75.25, "median": -75.25, "std": 0.1, "cv": 0.00133}
    p = write_ferebus_toml(
        tmp_path / "ferebus_O1.toml",
        system_name="WATER",
        natoms=3,
        atom="O1",
        alf=(1, 2, 3),
        target_prop=stats,
    )
    t = p.read_text(encoding="utf-8")
    for k in ("min", "max", "range", "mean", "median", "std", "cv"):
        assert ("target_prop_" + k + " = ") in t
    # the value round-trips, not just the key
    assert "target_prop_min = -75.5" in t


def test_ferebus_toml_omits_target_prop_when_absent(tmp_path):
    p = write_ferebus_toml(
        tmp_path / "ferebus.toml",
        system_name="X",
        natoms=2,
        atom="O1",
        alf=(1, 2, 3),
    )
    assert "target_prop_" not in p.read_text(encoding="utf-8")


def test_ferebus_toml_rejects_missing_atom_contract(tmp_path):
    try:
        write_ferebus_toml(
            tmp_path / "ferebus.toml",
            system_name="X",
            natoms=2,
            atom="",
            alf=(1, 2, 3),
        )
    except ValueError as exc:
        assert "atom is required" in str(exc)
    else:
        raise AssertionError("blank atom should reject")
