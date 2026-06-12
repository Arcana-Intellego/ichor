import numpy as np
import pytest

from ichor.hpc.active_learning.sampling.split import (
    AVAILABLE_STRATEGIES,
    SplitResult,
    get_split_strategy,
    split_pure_top_k,
    split_random,
    split_stratified_with_holdout,
)


def test_three_strategies_registered():
    assert set(AVAILABLE_STRATEGIES) == {
        "stratified_with_holdout",
        "random_80_20",
        "pure_top_k",
    }


def test_get_split_strategy_unknown_raises():
    with pytest.raises(ValueError):
        get_split_strategy("does_not_exist")


def test_get_split_strategy_returns_callable():
    fn = get_split_strategy("random_80_20")
    assert callable(fn)


def test_stratified_partition_is_disjoint_and_covers_all():
    alpha = [10.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.1]
    r = split_stratified_with_holdout(alpha, rng_seed=0)
    union = set(r.train_indices) | set(r.val_indices)
    assert union == set(range(len(alpha)))
    assert set(r.train_indices).isdisjoint(r.val_indices)


def test_stratified_holdout_is_subset_of_val_and_came_from_top_tier():
    alpha = list(range(20, 0, -1))   # descending so top-tier is [0..14] with default 0.75
    r = split_stratified_with_holdout(alpha, rng_seed=0)
    assert set(r.holdout_indices).issubset(r.val_indices)
    n_top_tier = int(round(0.75 * len(alpha)))
    top_tier_indices = set(np.argsort(-np.asarray(alpha), kind="stable")[:n_top_tier].tolist())
    assert set(r.holdout_indices).issubset(top_tier_indices)


def test_stratified_holdout_proportion_matches_request():
    alpha = list(range(100, 0, -1))
    r = split_stratified_with_holdout(alpha, train_fraction=0.75, high_holdout_fraction=0.10, rng_seed=0)
    n_top_tier = int(round(0.75 * 100))
    expected_holdout = int(round(0.10 * n_top_tier))
    assert len(r.holdout_indices) == expected_holdout


def test_stratified_holdout_cannot_empty_nonempty_training_set():
    r = split_stratified_with_holdout(
        [1.0],
        train_fraction=1.0,
        val_mid_fraction=0.0,
        high_holdout_fraction=1.0,
        rng_seed=0,
    )
    assert r.train_indices == [0]
    assert r.val_indices == []
    assert r.holdout_indices == []


def test_stratified_reproducible_with_same_seed():
    alpha = list(range(50, 0, -1))
    a = split_stratified_with_holdout(alpha, rng_seed=7)
    b = split_stratified_with_holdout(alpha, rng_seed=7)
    assert a.train_indices == b.train_indices
    assert a.val_indices == b.val_indices
    assert a.holdout_indices == b.holdout_indices


def test_stratified_validation_argument_ranges():
    with pytest.raises(ValueError):
        split_stratified_with_holdout([1.0], train_fraction=1.5)
    with pytest.raises(ValueError):
        split_stratified_with_holdout([1.0], val_mid_fraction=-0.1)
    with pytest.raises(ValueError):
        split_stratified_with_holdout([1.0], high_holdout_fraction=2.0)
    with pytest.raises(ValueError):
        split_stratified_with_holdout([1.0, 2.0], train_fraction=0.8, val_mid_fraction=0.5)


def test_random_split_default_fraction_is_80_20():
    alpha = list(range(10))
    r = split_random(alpha, rng_seed=42)
    assert r.n_train == 8
    assert r.n_val == 2


def test_random_split_reproducible_with_same_seed():
    alpha = list(range(30))
    a = split_random(alpha, rng_seed=11)
    b = split_random(alpha, rng_seed=11)
    assert a.train_indices == b.train_indices
    assert a.val_indices == b.val_indices


def test_pure_top_k_picks_highest_alpha_for_train():
    alpha = [10.0, 9.0, 5.0, 4.0, 1.0]
    r = split_pure_top_k(alpha, train_fraction=0.6)
    # n_train = round(0.6 * 5) = 3 -> top 3 alpha indices.
    assert set(r.train_indices) == {0, 1, 2}
    assert set(r.val_indices) == {3, 4}


def test_split_handles_empty_input():
    for strategy in (split_stratified_with_holdout, split_random, split_pure_top_k):
        r = strategy([])
        assert r.n_train == 0
        assert r.n_val == 0
        assert r.n_holdout == 0


def test_train_indices_sorted_ascending_for_all_strategies():
    alpha = list(np.random.default_rng(0).standard_normal(20))
    for strategy in (
        lambda a: split_stratified_with_holdout(a, rng_seed=0),
        lambda a: split_random(a, rng_seed=0),
        lambda a: split_pure_top_k(a),
    ):
        r = strategy(alpha)
        assert r.train_indices == sorted(r.train_indices)
        assert r.val_indices == sorted(r.val_indices)


# --- M15 F12: strategy label honesty -----------------------------------


def test_split_random_label_reflects_actual_fraction():
    from ichor.hpc.active_learning.sampling.split import (
        get_split_strategy,
        split_random,
    )
    result_70 = split_random([1.0] * 10, train_fraction=0.70, rng_seed=42)
    result_50 = split_random([1.0] * 10, train_fraction=0.50, rng_seed=42)
    # the recorded strategy stays the registry key so it round-trips; the
    # actual fraction rides along in metadata.
    assert result_70.strategy == "random_80_20"
    assert result_50.strategy == "random_80_20"
    assert result_70.metadata["train_fraction"] == 0.70
    assert result_50.metadata["train_fraction"] == 0.50
    assert get_split_strategy(result_70.strategy) is split_random


def test_split_random_label_default_fraction():
    from ichor.hpc.active_learning.sampling.split import split_random
    result = split_random([1.0] * 10, rng_seed=0)
    # Default train_fraction is 0.80
    assert result.strategy == "random_80_20"
    assert result.metadata["train_fraction"] == 0.80


def test_split_random_label_at_extremes():
    from ichor.hpc.active_learning.sampling.split import split_random
    result_0 = split_random([1.0] * 5, train_fraction=0.0, rng_seed=0)
    result_1 = split_random([1.0] * 5, train_fraction=1.0, rng_seed=0)
    assert result_0.strategy == "random_80_20"
    assert result_1.strategy == "random_80_20"
    assert result_0.metadata["train_fraction"] == 0.0
    assert result_1.metadata["train_fraction"] == 1.0
