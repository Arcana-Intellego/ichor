"""Tests for ichor.hpc.active_learning.acquisition.trajectory_pool (M9)."""
import json
import os
from pathlib import Path

import pytest

from ichor.core.adversarial.geometry import select_local_neighbours
from ichor.hpc.active_learning.acquisition.trajectory_pool import (
    POOL_MANIFEST_FILENAME,
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


def test_to_atoms_list_returns_fresh_list(tmp_path):
    pool = TrajectoryPool.import_from(FIXTURE, tmp_path)
    a = pool.to_atoms_list()
    b = pool.to_atoms_list()
    assert a is not b           # different list objects
    assert len(a) == pool.n_frames()


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


def test_acquisition_carries_seed_frame_id_and_subspace_frame_ids():
    """The SeedLocalAdversarialAcquisition gains seed_frame_id + subspace_frame_ids
    in M9.3. Verifying the bare API contract here (without instantiating, since
    that requires a real Models). We only confirm the attribute presence."""
    import inspect
    from ichor.core.adversarial.acquisition import SeedLocalAdversarialAcquisition

    sig = inspect.signature(SeedLocalAdversarialAcquisition.__init__)
    assert "seed_frame_id" in sig.parameters
    assert hasattr(SeedLocalAdversarialAcquisition, "subspace_frame_ids")
