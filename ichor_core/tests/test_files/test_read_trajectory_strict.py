from __future__ import annotations

import pytest

from ichor.core.files.xyz import Trajectory, XYZ
from ichor.core.files.xyz.strict_xyz import XYZParseError


def _read_trajectory(path):
    trajectory = Trajectory(path)
    trajectory.read()
    return trajectory


def test_trajectory_rejects_undeclared_atom_record(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text(
        "2\nframe 0\nH 0 0 0\nH 0 0 1\nO 9 9 9\n"
        "2\nframe 1\nH 0 0 0\nH 0 1 0\n",
        encoding="utf-8",
    )

    with pytest.raises(XYZParseError, match="line 5"):
        _read_trajectory(path)


@pytest.mark.parametrize(
    "body, message",
    [
        ("2\ncomment\nH 0 0 0\n", "truncated"),
        ("1\ncomment\nH nan 0 0\n", "non-finite"),
        ("1\ncomment\nNotAnElement 0 0 0\n", "unsupported element"),
        ("1 extra\ncomment\nH 0 0 0\n", "atom count"),
        ("1\n", "comment line"),
    ],
)
def test_trajectory_rejects_malformed_complete_grammar(tmp_path, body, message):
    path = tmp_path / "bad.xyz"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(XYZParseError, match=message):
        _read_trajectory(path)


def test_trajectory_accepts_blank_separators_and_extended_columns(tmp_path):
    path = tmp_path / "valid.xyz"
    path.write_text(
        "\n2\nframe 0\nH 0 0 0 extra\nH 0 0 1 extra\n\n"
        "2\nframe 1\nH 0 0 0\nH 0 1 0\n\n",
        encoding="utf-8",
    )

    trajectory = _read_trajectory(path)

    assert len(trajectory) == 2
    assert trajectory.atom_names == ["H1", "H2"]


def test_single_frame_reader_rejects_a_trajectory(tmp_path):
    path = tmp_path / "two.xyz"
    path.write_text(
        "1\nframe 0\nH 0 0 0\n1\nframe 1\nH 0 0 1\n",
        encoding="utf-8",
    )

    xyz = XYZ(path)
    with pytest.raises(ValueError, match="exactly one frame"):
        _ = xyz.atoms
