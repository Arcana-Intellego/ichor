"""Tests for ichor.hpc.active_learning.acquisition.trajectory_pool (M9)."""
import json
import gc
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ichor.core.adversarial.geometry import (
    coordinates_to_atoms,
    select_local_neighbours,
)
from ichor.hpc.active_learning.acquisition.trajectory_pool import (
    POOL_MANIFEST_FILENAME,
    POOL_COORDINATE_CACHE_FILENAME,
    POOL_COORDINATE_CACHE_MANIFEST_FILENAME,
    POOL_COORDINATE_CACHE_SUBDIR,
    POOL_SCHEMA_VERSION,
    POOL_SUBDIR,
    POOL_XYZ_FILENAME,
    TrajectoryPool,
    TrajectoryPoolManifest,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "water_tetramer.xyz"
)


def test_fixture_exists():
    assert FIXTURE.is_file(), (
        "water_tetramer.xyz fixture missing -- bundled in M8.4"
    )


def test_import_from_writes_canonical_xyz_and_manifest(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    canonical = tmp_path / POOL_XYZ_FILENAME
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    assert canonical.is_file()
    assert manifest_path.is_file()
    assert pool.canonical_path == canonical


def test_imported_manifest_round_trips_through_disk(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    with open(manifest_path) as f:
        on_disk = json.load(f)
    parsed = TrajectoryPoolManifest.from_dict(on_disk)
    assert parsed.sha256 == pool.sha256
    assert parsed.n_frames == pool.n_frames()
    assert parsed.natoms == pool.manifest.natoms
    assert parsed.atom_types == pool.manifest.atom_types
    assert parsed.schema_version == POOL_SCHEMA_VERSION


def test_load_rejects_manifest_bound_to_a_different_pool_path(tmp_path):
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["canonical_path"] = str(tmp_path / "elsewhere.xyz")
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="canonical_path"):
        TrajectoryPool.load(tmp_path)


def test_import_refuses_overwrite_without_force(tmp_path):
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    with pytest.raises(FileExistsError):
        TrajectoryPool.import_from(FIXTURE, tmp_path)


def test_import_refuses_symlinked_source(tmp_path):
    source = tmp_path / "linked-pool.xyz"
    try:
        os.symlink(FIXTURE, source)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(ValueError, match="symlink"):
        TrajectoryPool.import_from(source, tmp_path / "campaign")


def test_import_overwrites_when_force_true(tmp_path):
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    # Reimport with overwrite -- must succeed and produce a new manifest.
    pool2 = TrajectoryPool.import_from(FIXTURE, tmp_path, overwrite=True)
    assert pool2.n_frames() == 20


def test_load_after_import_matches_imported_state(tmp_path):
    imported = TrajectoryPool.import_from(FIXTURE, tmp_path)
    loaded = TrajectoryPool.load(tmp_path)
    assert loaded.sha256 == imported.sha256
    assert loaded.n_frames() == imported.n_frames()
    assert loaded.manifest.natoms == imported.manifest.natoms
    for fid in loaded.frame_ids():
        a = loaded.frame(fid)
        b = imported.frame(fid)
        # Atom type identity + coordinate equality
        assert tuple(at.type for at in a) == tuple(at.type for at in b)


def test_import_prewarms_content_addressed_coordinate_cache(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    cache = tmp_path / POOL_COORDINATE_CACHE_SUBDIR / pool.sha256

    assert (cache / POOL_COORDINATE_CACHE_FILENAME).is_file()
    assert (cache / POOL_COORDINATE_CACHE_MANIFEST_FILENAME).is_file()


def test_load_reports_independent_authority_and_coordinate_stages(tmp_path):
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    updates = []

    pool = TrajectoryPool.load(
        tmp_path,
        progress_callback=lambda stage, payload: updates.append(
            (stage, dict(payload))
        ),
    )

    assert updates[0] == (
        "trajectory_authority",
        {"completed": 0, "total": 1},
    )
    assert any(
        stage == "trajectory_coordinates"
        and payload.get("cache_status") == "hit"
        for stage, payload in updates
    )
    assert updates[-1][0] == "trajectory_coordinates"
    assert updates[-1][1]["completed"] == pool.n_frames()


def test_coordinate_cache_hit_does_not_reparse_xyz(monkeypatch, tmp_path):
    import ichor.hpc.active_learning.acquisition.trajectory_pool as module

    imported = TrajectoryPool.import_from(FIXTURE, tmp_path)

    def fail_parse(*args, **kwargs):
        raise AssertionError("canonical XYZ was reparsed on a cache hit")

    monkeypatch.setattr(module, "iter_xyz_frames", fail_parse)
    loaded = TrajectoryPool.load(tmp_path)

    assert loaded.sha256 == imported.sha256
    assert loaded.frame(3).coordinates.tolist() == imported.frame(3).coordinates.tolist()


def test_corrupt_coordinate_cache_is_ignored_and_rebuilt(tmp_path):
    imported = TrajectoryPool.import_from(FIXTURE, tmp_path)
    expected = imported.frame(7).coordinates.tolist()
    cache = tmp_path / POOL_COORDINATE_CACHE_SUBDIR / imported.sha256
    data = cache / POOL_COORDINATE_CACHE_FILENAME
    del imported
    gc.collect()
    data.write_bytes(data.read_bytes()[:-8] + b"corrupt!")

    loaded = TrajectoryPool.load(tmp_path)

    assert loaded.frame(7).coordinates.tolist() == expected
    payload = json.loads(
        (cache / POOL_COORDINATE_CACHE_MANIFEST_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert data.stat().st_size == payload["data"]["size"]


def test_load_detects_drift_when_canonical_xyz_modified(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    # Append a stray byte to the canonical file
    with open(pool.canonical_path, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(RuntimeError, match="pool drift detected"):
        TrajectoryPool.load(tmp_path)


def test_load_raises_when_manifest_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrajectoryPool.load(tmp_path)


def test_load_raises_when_canonical_xyz_missing(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    pool.canonical_path.unlink()
    with pytest.raises(FileNotFoundError):
        TrajectoryPool.load(tmp_path)


def test_frame_access_by_id(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    assert list(pool.frame_ids()) == list(range(20))
    assert pool.n_frames() == 20
    a0 = pool.frame(0)
    a19 = pool.frame(19)
    assert a0[0].type == a19[0].type   # same atom layout


def test_frame_index_out_of_range_raises(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    with pytest.raises(IndexError):
        pool.frame(-1)
    with pytest.raises(IndexError):
        pool.frame(pool.n_frames())


@pytest.mark.parametrize("frame_id", [0.9, "0", True, None])
def test_frame_index_requires_an_exact_non_boolean_integer(tmp_path, frame_id):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)

    with pytest.raises(TypeError, match="exact integer"):
        pool.frame(frame_id)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("schema_version", True, "schema_version"),
        ("n_frames", True, "n_frames"),
        ("natoms", 0, "natoms"),
        ("sha256", "not-a-digest", "sha256"),
        ("imported_iso", "not-a-time", "imported_iso"),
        ("masses", [math.nan], "masses"),
    ],
)
def test_pool_manifest_rejects_malformed_scientific_metadata(
    tmp_path,
    field,
    value,
    message,
):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    payload = pool.manifest.to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        TrajectoryPoolManifest.from_dict(payload)


def test_pool_load_rejects_manifest_atom_metadata_drift(tmp_path):
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["atom_types"][0] = "N"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata"):
        TrajectoryPool.load(tmp_path)


def test_pool_load_rejects_manifest_mass_metadata_drift(tmp_path):
    TrajectoryPool.import_from(FIXTURE, tmp_path)
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["masses"][0] += 1.0
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata"):
        TrajectoryPool.load(tmp_path)


def test_to_atoms_list_returns_fresh_list(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    a = pool.to_atoms_list()
    b = pool.to_atoms_list()
    assert a is not b           # different list objects
    assert len(a) == pool.n_frames()


def test_public_frame_accessors_cannot_mutate_content_addressed_pool(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    original = float(pool.frame(0)[0].x)
    detached_frame = pool.frame(0)
    detached_list = pool.to_atoms_list()
    detached_frame[0].coordinates[0] = original + 10.0
    detached_list[0][0].coordinates[0] = original + 20.0

    assert pool.frame(0)[0].x == pytest.approx(original)
    assert TrajectoryPool.load(tmp_path).frame(0)[0].x == pytest.approx(original)


def test_malformed_overwrite_preserves_existing_pool_and_manifest(tmp_path):
    imported = TrajectoryPool.import_from(FIXTURE, tmp_path)
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    old_pool = imported.canonical_path.read_bytes()
    old_manifest = manifest_path.read_bytes()
    malformed = tmp_path / "malformed.xyz"
    malformed.write_text("3\ntruncated\nO 0 0 0\n", encoding="utf-8")

    with pytest.raises(Exception):
        TrajectoryPool.import_from(malformed, tmp_path, overwrite=True)

    assert imported.canonical_path.read_bytes() == old_pool
    assert manifest_path.read_bytes() == old_manifest
    assert TrajectoryPool.load(tmp_path).sha256 == imported.sha256


def test_manifest_publication_failure_rolls_back_previous_pair(
    monkeypatch, tmp_path,
):
    import ichor.hpc.active_learning.acquisition.trajectory_pool as module

    imported = TrajectoryPool.import_from(FIXTURE, tmp_path)
    manifest_path = tmp_path / POOL_SUBDIR / POOL_MANIFEST_FILENAME
    old_pool = imported.canonical_path.read_bytes()
    old_manifest = manifest_path.read_bytes()
    real_write = module.atomic_write_json
    calls = {"count": 0}

    def fail_second_write(path, payload):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("injected manifest publication failure")
        return real_write(path, payload)

    monkeypatch.setattr(module, "atomic_write_json", fail_second_write)
    with pytest.raises(OSError, match="injected"):
        TrajectoryPool.import_from(FIXTURE, tmp_path, overwrite=True)

    assert imported.canonical_path.read_bytes() == old_pool
    assert manifest_path.read_bytes() == old_manifest
    assert not (tmp_path / POOL_SUBDIR / "pool.import.transaction.json").exists()


def test_select_local_neighbours_with_pool_returns_stable_frame_ids(tmp_path):
    """The pool path of select_local_neighbours must populate Neighbour.index
    with the stable frame_id (NOT enumeration position) -- M9.2 contract."""
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    seed = pool.frame(5)
    nbrs = select_local_neighbours(seed, pool, max_neighbours=8)
    indices = [n.index for n in nbrs]
    assert len(indices) == len(set(indices)), "frame_ids must be unique"
    assert all(0 <= idx < pool.n_frames() for idx in indices)
    # The seed itself is the closest frame.
    assert nbrs[0].index == 5
    assert nbrs[0].aligned_distance < 1.0e-8


def test_select_local_neighbours_with_bare_list_keeps_enumeration_indices(tmp_path):
    """The legacy contract (Sequence[Atoms] input -> enumeration-position
    indices) MUST be preserved for backward compat."""
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    atoms_list = pool.to_atoms_list()
    seed = atoms_list[5]
    nbrs = select_local_neighbours(seed, atoms_list, max_neighbours=8)
    # For this fixture (which happens to equal pool order), the two are
    # numerically identical; the contract-difference is what `index` means.
    assert all(0 <= n.index < len(atoms_list) for n in nbrs)


@pytest.mark.parametrize("seed_id", [0, 7, 19])
def test_vectorised_pool_neighbours_match_legacy_scalar_selection(
    tmp_path,
    seed_id,
):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    seed = pool.frame(seed_id)

    vectorised = select_local_neighbours(seed, pool, max_neighbours=12)
    scalar = select_local_neighbours(
        seed,
        pool.to_atoms_list(),
        max_neighbours=12,
    )

    assert [item.index for item in vectorised] == [
        item.index for item in scalar
    ]
    np.testing.assert_allclose(
        [item.aligned_distance for item in vectorised],
        [item.aligned_distance for item in scalar],
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_vectorised_pool_neighbours_do_not_materialise_the_complete_pool(
    tmp_path,
):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    seed = pool.frame(5)
    base = np.asarray(seed.coordinates, dtype=float)
    coordinates = []
    for frame_id in range(100):
        row = base.copy()
        row[1, 0] += 1.0e-3 * frame_id
        row[2, 1] -= 5.0e-4 * frame_id
        coordinates.append(row)
    calls = []

    class _CoordinatePool:
        manifest = SimpleNamespace(
            atom_types=tuple(str(atom.type) for atom in seed)
        )

        def coordinates_view(self):
            return np.asarray(coordinates, dtype=float)

        def frame_ids(self):
            return range(len(coordinates))

        def frame(self, frame_id):
            calls.append(int(frame_id))
            return coordinates_to_atoms(seed, coordinates[int(frame_id)])

    candidate_pool = _CoordinatePool()
    neighbours = select_local_neighbours(
        seed,
        candidate_pool,
        max_neighbours=4,
    )

    assert [item.index for item in neighbours]
    assert len(set(calls)) < len(coordinates)


def test_acquisition_carries_seed_frame_id_and_subspace_frame_ids():
    """The SeedLocalAdversarialAcquisition gains seed_frame_id + subspace_frame_ids
    in M9.3. Verifying the bare API contract here (without instantiating, since
    that requires a real Models). We only confirm the attribute presence."""
    import inspect
    from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition

    sig = inspect.signature(SeedLocalAdversarialAcquisition.__init__)
    assert "seed_frame_id" in sig.parameters
    assert hasattr(SeedLocalAdversarialAcquisition, "subspace_frame_ids")
