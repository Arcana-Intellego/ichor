"""M15 F10 tests: provenance read-modify-write paths are flock-protected.

Spawn multiple workers concurrently against the same provenance sidecar,
index file, or recent_seeds file and assert every write lands. Pre-F10
these tests would lose records to the load-modify-store race; post-F10
they pass because each RMW body holds an exclusive flock for its duration.
"""
import json
import threading
from pathlib import Path

import pytest

from ichor.hpc.active_learning.versioning.provenance import (
    append_recent_seeds,
    append_to_index,
    enrich_with_anti_overlap,
    enrich_with_ariadne,
    enrich_with_phase_b,
    load_index,
    load_recent_seeds_payload,
    read_provenance,
    write_seed_provenance,
)


def _setup_pointdir(tmp_path, seed_frame_id=0):
    pdir = tmp_path / "POINT_0001.pointdir"
    write_seed_provenance(
        pdir,
        campaign_uid="conc-test",
        iteration=0,
        trajectory_sha256="abcd",
        seed_frame_id=seed_frame_id,
        seed_selection_origin="bulk",
        seed_variance_at_selection=None,
        subspace_neighbour_frame_ids=[seed_frame_id],
        subspace_dimension=1,
        subspace_eigenvalues=[1.0],
    )
    return pdir


def test_concurrent_enriches_dont_clobber(tmp_path):
    """Four threads each call a different enrich_with_* on the SAME pointdir.
    After all join, every section must be present (pre-F10: races would
    leave one or two sections silently missing)."""
    pdir = _setup_pointdir(tmp_path)

    def worker_ariadne():
        for _ in range(8):
            enrich_with_ariadne(
                pdir, alpha_initial=0.1, alpha_final=0.5,
                n_evaluations=50, fell_back_to_ds=False, wall_seconds=12.0,
            )

    def worker_anti_overlap():
        for _ in range(8):
            enrich_with_anti_overlap(
                pdir, min_whitened_distance_to_training=0.5,
                passed=True, flag=None,
            )

    def worker_phase_b():
        for _ in range(8):
            enrich_with_phase_b(
                pdir, selected_after_fps=True, diversity_rank=3,
                descriptor_used="hybrid_alf_rmsd",
            )

    threads = [
        threading.Thread(target=worker_ariadne),
        threading.Thread(target=worker_anti_overlap),
        threading.Thread(target=worker_phase_b),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = read_provenance(pdir)
    assert data["ariadne"] is not None
    assert data["anti_overlap"] is not None
    assert data["phase_b"] is not None
    # And the values are the last-write-wins of each section (deterministic
    # because each worker wrote the same payload many times).
    assert data["ariadne"]["alpha_final"] == 0.5
    assert data["anti_overlap"]["passed"] is True
    assert data["phase_b"]["diversity_rank"] == 3


def test_concurrent_index_appends_persist_all(tmp_path):
    """Eight workers each append 25 records to the same index file. After
    join, the index must contain exactly 8 * 25 = 200 records (pre-F10: a
    handful would be lost to load-modify-store races)."""
    N_WORKERS = 8
    N_PER_WORKER = 25

    def worker(worker_id):
        for i in range(N_PER_WORKER):
            append_to_index(
                tmp_path,
                iteration=worker_id,
                pointdir_name=f"POINT_w{worker_id}_i{i:04d}.pointdir",
                seed_frame_id=worker_id * 100 + i,
            )

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = load_index(tmp_path)
    assert len(data["records"]) == N_WORKERS * N_PER_WORKER
    # And every (worker_id, i) pair must be present.
    pairs = {(r["iteration"], r["pointdir_name"]) for r in data["records"]}
    expected = {
        (w, f"POINT_w{w}_i{i:04d}.pointdir")
        for w in range(N_WORKERS) for i in range(N_PER_WORKER)
    }
    assert pairs == expected


def test_concurrent_recent_seeds_appends_no_loss(tmp_path):
    """Two workers race against recent_seeds.json. The trim semantics make
    final ordering non-deterministic, but every successful append must
    leave the file readable and structurally valid (no corrupted JSON)."""
    N_PER_WORKER = 30

    def worker(worker_id):
        for i in range(N_PER_WORKER):
            append_recent_seeds(
                tmp_path,
                iteration=worker_id * 1000 + i,
                frame_ids=[worker_id * 1000 + i],
                cooldown=10,
            )

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The file must still be valid JSON with schema_version + history.
    data = load_recent_seeds_payload(tmp_path)
    assert data["schema_version"] == 1
    assert isinstance(data["history"], list)
    # With cooldown=10 the history is capped at 10 entries.
    assert len(data["history"]) <= 10


def test_lock_file_does_not_interfere_with_provenance_read(tmp_path):
    """After enrich_with_*, the read_provenance call must still work even
    though a sibling .provenance.lock file is present in the pointdir."""
    pdir = _setup_pointdir(tmp_path)
    enrich_with_ariadne(
        pdir, alpha_initial=0.0, alpha_final=1.0,
        n_evaluations=1, fell_back_to_ds=False, wall_seconds=0.1,
    )
    # The lock file should exist alongside the provenance.
    assert (pdir / ".provenance.lock").exists()
    # And the provenance is readable as usual.
    data = read_provenance(pdir)
    assert data["ariadne"]["alpha_final"] == 1.0
