from types import SimpleNamespace

import numpy as np

from ichor.hpc.active_learning.daemon.seed_selection_runtime import (
    SeedSelectionProgressReporter,
    SeedSelectionRuntimeCache,
    finalise_seed_selection_workspace,
)


class _Pool:
    def __init__(self, n_frames=12):
        self.sha256 = "a" * 64
        self.manifest = SimpleNamespace(
            natoms=1,
            atom_types=("O",),
            masses=(15.999,),
        )
        self._n_frames = int(n_frames)

    def n_frames(self):
        return self._n_frames

    def frame(self, frame_id):
        return int(frame_id)


class _Models:
    ialf_dict = {"O1": np.asarray([0, 0, 0], dtype=int)}

    def __init__(self):
        self.feature_calls = 0

    def get_features_dict(self, frame):
        self.feature_calls += 1
        value = float(frame)
        return {"O1": np.asarray([value, value + 0.5], dtype=float)}


class _Posterior:
    property_name = "iqa"
    scaled = True

    def __init__(self):
        self.models = _Models()
        self._property_models = {
            "O1": SimpleNamespace(nfeats=2, ntrain=3),
        }
        self.variance_calls = 0
        self.prepare_calls = 0
        self.projection_blocks = []
        self.interrupt_projection_after = None

    def variances_from_feature_arrays(
        self, feature_arrays, *, chunk_size=None, progress_callback=None
    ):
        del chunk_size
        self.variance_calls += 1
        values = np.sum(np.asarray(feature_arrays["O1"]) ** 2, axis=1)
        if progress_callback is not None:
            progress_callback(len(values), len(values))
        return values

    def prepare_feature_batch(
        self,
        feature_arrays,
        *,
        row_ids,
        projection_directory,
        max_resident_projection_bytes,
        projection_resume_columns=None,
        projection_progress_callback=None,
    ):
        del max_resident_projection_bytes
        self.prepare_calls += 1
        ids = np.asarray(row_ids, dtype=np.int64)
        projection_directory.mkdir(parents=True, exist_ok=True)
        projections = {}
        for position, (atom, model) in enumerate(self._property_models.items()):
            path = projection_directory / (
                "projection-" + str(position).zfill(4) + ".npy"
            )
            completed = int((projection_resume_columns or {}).get(atom, 0))
            projection = np.lib.format.open_memmap(
                path,
                mode="r+" if completed else "w+",
                **(
                    {}
                    if completed
                    else {
                        "dtype": np.float64,
                        "shape": (int(model.ntrain), int(ids.size)),
                    }
                ),
            )
            for start in range(completed, int(ids.size), 2):
                stop = min(int(ids.size), start + 2)
                projection[:, start:stop] = np.arange(
                    start, stop, dtype=float
                )[None, :]
                projection.flush()
                self.projection_blocks.append((str(atom), int(start), int(stop)))
                if projection_progress_callback is not None:
                    projection_progress_callback(str(atom), int(start), int(stop))
                if (
                    self.interrupt_projection_after is not None
                    and len(self.projection_blocks)
                    >= int(self.interrupt_projection_after)
                ):
                    raise RuntimeError("injected projection interruption")
            projections[atom] = projection
        return SimpleNamespace(
            row_ids=ids,
            features=feature_arrays,
            projections=projections,
            means=np.asarray(ids, dtype=np.float64),
            variances=np.asarray(ids, dtype=np.float64) + 1.0,
            signal_variances={atom: 1.0 for atom in projections},
        )


def _cache(tmp_path, posterior, progress=None):
    return SeedSelectionRuntimeCache(
        tmp_path / "campaign",
        pool=_Pool(),
        posterior=posterior,
        model_set_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
        iteration=2,
        progress=progress,
    )


def test_feature_and_variance_caches_are_reused_without_reextracting(tmp_path):
    first_posterior = _Posterior()
    first = _cache(tmp_path, first_posterior)
    indexed = first.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )

    assert first_posterior.models.feature_calls == 12
    assert first_posterior.variance_calls == 3
    expected = np.asarray(
        [float(index) ** 2 + (float(index) + 0.5) ** 2 for index in range(12)]
    )
    np.testing.assert_allclose(indexed.variances_by_index(range(12)), expected)

    second_posterior = _Posterior()
    second = _cache(tmp_path, second_posterior)
    reused = second.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )

    assert second_posterior.models.feature_calls == 0
    assert second_posterior.variance_calls == 0
    np.testing.assert_allclose(reused.variances_by_index(range(12)), expected)


def test_corrupt_feature_cache_is_ignored_and_rebuilt(tmp_path):
    import gc

    posterior = _Posterior()
    cache = _cache(tmp_path, posterior)
    arrays, manifest = cache.ensure_features()
    feature_dir = cache.root / "features" / manifest["cache_id"]
    del arrays, cache
    gc.collect()
    array_path = next(feature_dir.glob("atom-*.npy"))
    with array_path.open("r+b") as handle:
        handle.seek(-8, 2)
        handle.write(b"\x00" * 8)

    rebuilt_posterior = _Posterior()
    rebuilt = _cache(tmp_path, rebuilt_posterior)
    rebuilt.ensure_features()

    assert rebuilt_posterior.models.feature_calls == 12


def test_corrupt_variance_cache_is_ignored_and_rebuilt(tmp_path):
    import gc

    posterior = _Posterior()
    cache = _cache(tmp_path, posterior)
    indexed = cache.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    variance_path = (
        cache.root / "variances" / (indexed.variance_cache_id + ".npy")
    )
    del indexed, cache
    gc.collect()
    with variance_path.open("r+b") as handle:
        handle.seek(-8, 2)
        handle.write(b"\xff" * 8)

    rebuilt_posterior = _Posterior()
    rebuilt = _cache(tmp_path, rebuilt_posterior)
    rebuilt.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )

    assert rebuilt_posterior.variance_calls == 3


def test_partial_feature_build_resumes_from_last_verified_chunk(tmp_path):
    class _InterruptedModels(_Models):
        def get_features_dict(self, frame):
            if int(frame) == 270:
                raise RuntimeError("injected interruption")
            return super().get_features_dict(frame)

    campaign = tmp_path / "campaign"
    interrupted_posterior = _Posterior()
    interrupted_posterior.models = _InterruptedModels()
    interrupted = SeedSelectionRuntimeCache(
        campaign,
        pool=_Pool(n_frames=300),
        posterior=interrupted_posterior,
        model_set_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
        iteration=2,
    )

    import pytest

    with pytest.raises(RuntimeError, match="injected interruption"):
        interrupted.ensure_features()

    resumed_posterior = _Posterior()
    resumed = SeedSelectionRuntimeCache(
        campaign,
        pool=_Pool(n_frames=300),
        posterior=resumed_posterior,
        model_set_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
        iteration=2,
    )
    resumed.ensure_features()

    assert resumed_posterior.models.feature_calls == 44


def test_completed_projection_workspace_is_reused_after_restart(
    tmp_path, monkeypatch
):
    from ichor.hpc.active_learning.daemon import seed_selection_runtime as runtime

    monkeypatch.setattr(runtime, "MAX_RESIDENT_PROJECTION_BYTES", 1)
    first_posterior = _Posterior()
    first = _cache(tmp_path, first_posterior)
    indexed = first.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    prepared = indexed.prepare_indexed_batch([1, 3, 5, 7])
    assert first_posterior.prepare_calls == 1
    np.testing.assert_array_equal(prepared.means, np.asarray([1.0, 3.0, 5.0, 7.0]))

    second_posterior = _Posterior()
    second = _cache(tmp_path, second_posterior)
    reused = second.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )

    def _must_not_prepare(*args, **kwargs):
        raise AssertionError("completed projection workspace should be reused")

    second_posterior.prepare_feature_batch = _must_not_prepare
    restored = reused.prepare_indexed_batch([1, 3, 5, 7])

    np.testing.assert_array_equal(
        restored.means_by_index([1, 3, 5, 7]),
        np.asarray([1.0, 3.0, 5.0, 7.0]),
    )


def test_corrupt_projection_workspace_is_rebuilt(tmp_path, monkeypatch):
    from ichor.hpc.active_learning.daemon import seed_selection_runtime as runtime

    monkeypatch.setattr(runtime, "MAX_RESIDENT_PROJECTION_BYTES", 1)
    first_posterior = _Posterior()
    first = _cache(tmp_path, first_posterior)
    indexed = first.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    indexed.prepare_indexed_batch([1, 3, 5, 7])
    indexed.close()
    means_path = first.projection_directory.parent / "means.npy"
    with means_path.open("r+b") as handle:
        handle.seek(-8, 2)
        handle.write(b"\xff" * 8)

    rebuilt_posterior = _Posterior()
    rebuilt = _cache(tmp_path, rebuilt_posterior)
    rebuilt_indexed = rebuilt.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    rebuilt_indexed.prepare_indexed_batch([1, 3, 5, 7])

    assert rebuilt_posterior.prepare_calls == 1


def test_partial_variance_build_resumes_from_verified_chunk(tmp_path):
    class _InterruptedPosterior(_Posterior):
        def variances_from_feature_arrays(self, *args, **kwargs):
            if self.variance_calls == 1:
                raise RuntimeError("injected variance interruption")
            return super().variances_from_feature_arrays(*args, **kwargs)

    import pytest

    interrupted_posterior = _InterruptedPosterior()
    interrupted = _cache(tmp_path, interrupted_posterior)
    with pytest.raises(RuntimeError, match="injected variance interruption"):
        interrupted.ensure_indexed_posterior(
            eligible_indices=list(range(12)),
            chunk_size=4,
        )

    resumed_posterior = _Posterior()
    resumed = _cache(tmp_path, resumed_posterior)
    indexed = resumed.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )

    assert resumed_posterior.variance_calls == 2
    assert indexed.variances_by_index(range(12)).shape == (12,)


def test_partial_projection_workspace_resumes_from_verified_block(
    tmp_path, monkeypatch
):
    import pytest
    from ichor.hpc.active_learning.daemon import seed_selection_runtime as runtime

    monkeypatch.setattr(runtime, "MAX_RESIDENT_PROJECTION_BYTES", 1)
    interrupted_posterior = _Posterior()
    interrupted_posterior.interrupt_projection_after = 1
    interrupted = _cache(tmp_path, interrupted_posterior)
    indexed = interrupted.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    with pytest.raises(RuntimeError, match="injected projection interruption"):
        indexed.prepare_indexed_batch([1, 3, 5, 7])

    resumed_posterior = _Posterior()
    resumed = _cache(tmp_path, resumed_posterior)
    resumed_indexed = resumed.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    prepared = resumed_indexed.prepare_indexed_batch([1, 3, 5, 7])

    assert resumed_posterior.projection_blocks[0] == ("O1", 2, 4)
    np.testing.assert_array_equal(
        prepared.means,
        np.asarray([1.0, 3.0, 5.0, 7.0]),
    )


def test_successful_selection_prunes_superseded_cache_namespaces(tmp_path):
    posterior = _Posterior()
    cache = _cache(tmp_path, posterior)
    indexed = cache.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
    )
    stale_paths = [
        cache.root / "features" / ".building-stale",
        cache.root / "features" / "stale-feature",
        cache.root / "variances" / ".building-stale",
        cache.root / "workspaces" / "iteration-000001",
    ]
    for path in stale_paths:
        path.mkdir(parents=True, exist_ok=True)
        (path / "stale").write_text("stale", encoding="utf-8")
    neighbour = cache.root / "neighbours" / "stale.json"
    neighbour.parent.mkdir(parents=True, exist_ok=True)
    neighbour.write_text("{}", encoding="utf-8")
    cache.active_neighbour_cache_id = "active-neighbour"

    cache.prune_superseded(
        active_feature_cache_id=indexed.feature_cache_id,
        active_variance_cache_id=indexed.variance_cache_id,
    )

    assert all(not path.exists() for path in stale_paths)
    assert not neighbour.exists()
    assert (cache.root / "features" / indexed.feature_cache_id).is_dir()
    assert (
        cache.root / "variances" / (indexed.variance_cache_id + ".npy")
    ).is_file()


def test_published_selection_reentry_removes_iteration_workspace(tmp_path):
    campaign = tmp_path / "campaign"
    workspace = (
        campaign
        / ".DATA"
        / "CACHE"
        / "SEED_SELECT"
        / "workspaces"
        / "iteration-000002"
    )
    workspace.mkdir(parents=True)
    (workspace / "partial.npy").write_bytes(b"partial")

    finalise_seed_selection_workspace(campaign, iteration=2)

    assert not workspace.exists()


def test_deferred_posterior_draws_random_subset_before_model_scoring(tmp_path):
    from ichor.hpc.active_learning.acquisition.seed_selection import select_seeds

    posterior = _Posterior()
    cache = _cache(tmp_path, posterior)
    cache.ensure_features()
    indexed = cache.ensure_indexed_posterior(
        eligible_indices=list(range(12)),
        chunk_size=4,
        defer_variances=True,
    )
    observations = []

    def _progress(stage, payload):
        observations.append((str(stage), int(posterior.variance_calls), dict(payload)))

    result = select_seeds(
        list(range(12)),
        indexed,
        n_seeds=4,
        bulk_fraction=0.5,
        rng_seed=91,
        strategy="hybrid_variance",
        progress_callback=_progress,
    )

    assert result.n == 4
    random_events = [item for item in observations if item[0] == "random"]
    assert random_events
    assert all(variance_calls == 0 for _, variance_calls, _ in random_events)
    assert posterior.variance_calls == 3


def test_progress_reporter_publishes_current_stage_and_journal_events(tmp_path):
    events = []

    def _journal(event, **payload):
        events.append((event, payload))

    reporter = SeedSelectionProgressReporter(
        tmp_path / "campaign",
        campaign_uid="campaign-1",
        iteration=2,
        journal_event=_journal,
    )
    reporter.bind_inputs(
        trajectory_sha256="a" * 64,
        model_set_sha256="b" * 64,
        model_manifest_sha256="c" * 64,
    )
    reporter.update("variance", force=True, completed=7, total=12)

    from ichor.hpc.active_learning.strict_json import strict_json as json

    payload = json.loads(reporter.path.read_text(encoding="utf-8"))
    assert payload["stage"] == "variance"
    assert payload["completed"] == 7
    assert payload["total"] == 12
    assert payload["daemon_start_identity"]
    assert payload["trajectory_sha256"] == "a" * 64
    assert payload["model_set_sha256"] == "b" * 64
    assert payload["model_manifest_sha256"] == "c" * 64
    assert [event for event, _payload in events] == [
        "seed_selection_started",
        "seed_selection_progress",
        "seed_selection_progress",
    ]


def test_progress_reporter_throttles_same_stage_journal_updates(tmp_path):
    events = []
    reporter = SeedSelectionProgressReporter(
        tmp_path / "campaign",
        campaign_uid="campaign-1",
        iteration=2,
        journal_event=lambda event, **payload: events.append((event, payload)),
    )
    reporter.update("variance", force=True, completed=1, total=12)
    count_after_stage_change = len(events)
    reporter.update("variance", force=True, completed=2, total=12)
    assert len(events) == count_after_stage_change

    reporter._last_journal -= 31.0
    reporter.update("variance", force=True, completed=3, total=12)
    assert len(events) == count_after_stage_change + 1


def test_status_uses_current_seed_selection_progress(tmp_path):
    from ichor.hpc.active_learning import cli
    from ichor.hpc.active_learning.daemon.state import CampaignPhase

    campaign = tmp_path / "campaign"
    reporter = SeedSelectionProgressReporter(
        campaign,
        campaign_uid="campaign-1",
        iteration=2,
    )
    reporter.update(
        "d_optimal",
        force=True,
        completed=73,
        total=160,
        shortlist_size=1280,
    )
    state = SimpleNamespace(
        campaign_uid="campaign-1",
        iteration=2,
        phase=CampaignPhase.SEED_SELECT,
    )
    runtime = {
        "phase": CampaignPhase.SEED_SELECT.value,
        "lock_held": True,
        "lease_dir_exists": False,
        "background_pid": None,
        "background_pid_alive": False,
        "active_submission_intents": [],
        "pending_jobs": {},
    }

    progress = cli._load_seed_selection_progress_status(campaign, state, runtime)
    assert progress["state"] == "current"
    runtime["seed_selection_progress"] = progress
    assert cli._status_current_activity(runtime) == (
        "Selecting D-optimal seeds 73/160 from 1280 shortlisted frames."
    )

    assert cli._format_seed_selection_progress(
        {
            "stage": "reference_scales",
            "sample": 14,
            "samples": 24,
            "reference_pass": "tuned",
        }
    ) == "Computing reference scales: sample 14/24, tuned stencils."

    reporter.finish(selection_published=True)
    completed = cli._load_seed_selection_progress_status(campaign, state, runtime)
    assert completed["state"] == "current"
    assert cli._format_seed_selection_progress(completed["record"]) == (
        "Seed selection has been published."
    )

    stale_state = SimpleNamespace(
        campaign_uid="campaign-1",
        iteration=3,
        phase=CampaignPhase.SEED_SELECT,
    )
    stale = cli._load_seed_selection_progress_status(
        campaign,
        stale_state,
        runtime,
    )
    assert stale["state"] == "stale"


def test_status_ignores_malformed_seed_selection_progress_by_default(tmp_path):
    from ichor.hpc.active_learning import cli
    from ichor.hpc.active_learning.daemon.state import CampaignPhase

    campaign = tmp_path / "campaign"
    path = (
        campaign
        / ".DATA"
        / "ACTIVE_LEARNING"
        / "runtime_progress"
        / "SEED_SELECT.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    state = SimpleNamespace(
        campaign_uid="campaign-1",
        iteration=2,
        phase=CampaignPhase.SEED_SELECT,
    )
    runtime = {
        "phase": CampaignPhase.SEED_SELECT.value,
        "lock_held": True,
        "lease_dir_exists": False,
        "background_pid_alive": False,
        "active_submission_intents": [],
        "pending_jobs": {},
    }

    progress = cli._load_seed_selection_progress_status(campaign, state, runtime)
    assert progress["state"] == "malformed"
    runtime["seed_selection_progress"] = progress
    assert cli._status_current_activity(runtime) == (
        "The daemon is working on ARIADNE seed selection."
    )
