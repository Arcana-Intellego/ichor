from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ichor.hpc.active_learning.sampling import phase_b_reference as cache


@dataclass(frozen=True)
class _Entry:
    global_ordinal: int
    pointdir_path: Path

    def identity_payload(self):
        return {
            "global_ordinal": int(self.global_ordinal),
            "pointdir": self.pointdir_path.name,
        }


@dataclass(frozen=True)
class _View:
    campaign_uid: str
    version: int
    cumulative_view_sha256: str
    head_manifest_sha256: str
    entries: tuple[_Entry, ...]


def _pointdir(root: Path, ordinal: int, shift: float) -> _Entry:
    pointdir = root / ("POINT_" + f"{ordinal:04d}" + ".pointdir")
    pointdir.mkdir(parents=True)
    xyz = pointdir / ("POINT_" + f"{ordinal:04d}" + ".xyz")
    xyz.write_text(
        "2\npoint\n"
        + "H " + str(float(shift)) + " 0.0 0.0\n"
        + "H " + str(float(shift + 1.0)) + " 0.0 0.0\n",
        encoding="utf-8",
    )
    return _Entry(global_ordinal=ordinal, pointdir_path=pointdir.resolve())


def _view(campaign: Path) -> _View:
    entries = (
        _pointdir(campaign / "QM_REFERENCE_DATA" / "points", 0, 0.0),
        _pointdir(campaign / "QM_REFERENCE_DATA" / "points", 1, 0.25),
    )
    return _View(
        campaign_uid="campaign",
        version=0,
        cumulative_view_sha256="a" * 64,
        head_manifest_sha256="b" * 64,
        entries=entries,
    )


def test_phase_b_reference_cache_hit_does_not_reparse_pointdirs(
    tmp_path,
    monkeypatch,
):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    view = _view(campaign)

    first = cache.load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        workers=2,
    )
    expected = np.asarray(first.coordinates).copy()
    assert first.n_references == 2
    assert first.cache_status == "hit"

    monkeypatch.setattr(
        cache,
        "_parse_reference_entry",
        lambda _entry: (_ for _ in ()).throw(AssertionError("cold parse")),
    )
    second = cache.load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        workers=2,
    )

    assert second.cache_status == "hit"
    np.testing.assert_array_equal(second.coordinates, expected)


def test_phase_b_reference_cache_quarantines_corruption_and_republishes(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    view = _view(campaign)
    first = cache.load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        workers=2,
    )
    expected = np.asarray(first.coordinates).copy()
    paths = cache._cache_paths(campaign.resolve(), view)
    del first
    gc.collect()
    paths["data"].write_bytes(paths["data"].read_bytes() + b"corrupt")

    rebuilt = cache.load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        workers=2,
    )

    assert rebuilt.cache_status == "hit"
    np.testing.assert_array_equal(rebuilt.coordinates, expected)
    retired = list(paths["root"].glob(".bad-*"))
    assert len(retired) == 1
    assert paths["directory"].is_dir()


def test_phase_b_reference_cache_rejects_external_source_path(tmp_path):
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    view = _view(campaign)
    cache.load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        workers=2,
    )
    paths = cache._cache_paths(campaign.resolve(), view)
    payload = cache.json.loads(paths["manifest"].read_text(encoding="utf-8"))
    payload["sources"][0]["source"] = str((tmp_path / "outside.xyz").resolve())
    cache.atomic_write_json(paths["manifest"], payload)

    rebuilt = cache.load_phase_b_reference_coordinates(
        campaign,
        reference_view=view,
        workers=2,
    )

    np.testing.assert_allclose(rebuilt.coordinates[0, 1, 0], 1.0)
    assert rebuilt.cache_status == "hit"
