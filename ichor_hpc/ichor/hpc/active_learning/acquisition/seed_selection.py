"""Seed selection for the adversarial attack phase.

By default, part of the batch is sampled uniformly at random from the current
training set and the remainder is chosen by top posterior variance. Operators
can opt into a cheap D-optimal mode for the non-random part: it still starts
from high-variance candidates, but greedily avoids points that are redundant
with seeds already selected in model-posterior covariance space.

Current wiring: callers may pass "training_frame_ids" (parallel to
"training_atoms") that label each training point with its stable trajectory
frame_id, plus "forbidden_frame_ids" listing frames that must be excluded
(union of training-pool provenance and the recent-seeds cooldown cache).
:class:"SeedSelection" carries "frame_ids" for the picked positions so the
provenance ledger can record exactly which MD frames seeded which committed
pointdir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from ichor.core.atoms import Atoms


__all__ = ["SeedSelection", "select_seeds"]


VALID_STRATEGIES = frozenset({"hybrid_variance", "d_optimal"})
RANKING_SCORE_QUANTISATION = 1.0e-12


def _finite_variances(values, *, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(context + " contains non-finite posterior variance")
    return arr


def _finite_matrix(values, *, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(context + " must be a 2D covariance matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(context + " contains non-finite posterior covariance")
    return arr


def _quantised_descending_order(
    scores,
    indices,
    *,
    quantum: float = RANKING_SCORE_QUANTISATION,
) -> np.ndarray:
    """Deterministic descending score order with candidate-index tie-breaks."""
    arr = _finite_variances(scores, context="seed ranking score")
    idx = np.asarray(indices, dtype=int)
    if arr.shape != idx.shape:
        raise ValueError("seed ranking scores and indices length mismatch")
    raw_order = np.lexsort((idx, -arr))
    if float(quantum) <= 0.0 or raw_order.size <= 1:
        return raw_order
    grouped: List[int] = []
    start = 0
    while start < raw_order.size:
        stop = start + 1
        group_max = float(arr[int(raw_order[start])])
        while (
            stop < raw_order.size
            and group_max - float(arr[int(raw_order[stop])]) <= float(quantum)
        ):
            stop += 1
        group = list(raw_order[start:stop])
        group.sort(key=lambda pos: int(idx[int(pos)]))
        grouped.extend(int(pos) for pos in group)
        start = stop
    return np.asarray(grouped, dtype=int)


def _quantised_tie_count(
    scores,
    *,
    quantum: float = RANKING_SCORE_QUANTISATION,
) -> int:
    arr = _finite_variances(scores, context="seed ranking score")
    if arr.size <= 1:
        return 0
    order = _quantised_descending_order(arr, np.arange(arr.size), quantum=0.0)
    ties = 0
    start = 0
    while start < order.size:
        stop = start + 1
        group_max = float(arr[int(order[start])])
        while (
            stop < order.size
            and group_max - float(arr[int(order[stop])]) <= float(quantum)
        ):
            stop += 1
        if stop - start > 1:
            ties += stop - start
        start = stop
    return int(ties)


def _posterior_variances(posterior, points, *, chunk_size: Optional[int]):
    if not hasattr(posterior, "variances"):
        return [posterior.variance(point) for point in points]
    if chunk_size is None:
        return posterior.variances(points)
    try:
        return posterior.variances(points, chunk_size=int(chunk_size))
    except TypeError as exc:
        # Compatibility with older tests and light-weight posterior adapters
        # that implement variances(points) but not the new chunk-size keyword.
        if "chunk_size" not in str(exc):
            raise
        return posterior.variances(points)


def _posterior_cross_covariances(
    posterior,
    left_points,
    right_points,
    *,
    chunk_size: Optional[int],
) -> np.ndarray:
    """Return covariance(left, right) without asking for a full pool matrix."""
    n_left = len(left_points)
    n_right = len(right_points)
    if n_left == 0 or n_right == 0:
        return np.zeros((n_left, n_right), dtype=float)
    if hasattr(posterior, "cross_covariances"):
        try:
            return _finite_matrix(
                posterior.cross_covariances(
                    left_points,
                    right_points,
                    chunk_size=chunk_size,
                ),
                context="D-optimal posterior cross-covariance",
            )
        except TypeError as exc:
            if "chunk_size" not in str(exc):
                raise
            return _finite_matrix(
                posterior.cross_covariances(left_points, right_points),
                context="D-optimal posterior cross-covariance",
            )
    if not hasattr(posterior, "covariance"):
        raise ValueError(
            "d_optimal seed_selection.strategy requires posterior.covariance "
            "or posterior.cross_covariances"
        )
    cov = np.zeros((n_left, n_right), dtype=float)
    for i, left in enumerate(left_points):
        for j, right in enumerate(right_points):
            cov[i, j] = float(posterior.covariance(left, right))
    return _finite_matrix(cov, context="D-optimal posterior cross-covariance")


def _d_optimal_select(
    *,
    training_atoms: Sequence[Atoms],
    posterior,
    remaining_indices: Sequence[int],
    remaining_variances: np.ndarray,
    remaining_scores: np.ndarray,
    selected_context: Sequence[int],
    n_select: int,
    pool_multiplier: int,
    jitter: float,
    novelty_floor: float,
    score_power: float,
    variance_chunk_size: Optional[int],
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, Any]]:
    """Greedy D-optimal pick from a bounded high-variance candidate pool."""
    if n_select <= 0 or not remaining_indices:
        return [], [], {
            "prefilter_pool_size": 0,
            "d_optimal_requested": int(max(0, n_select)),
            "d_optimal_selected": 0,
        }
    n_pool = min(
        len(remaining_indices),
        max(int(n_select), int(n_select) * int(pool_multiplier)),
    )
    order = _quantised_descending_order(remaining_scores, remaining_indices)
    pool_positions = [int(pos) for pos in order[:n_pool]]
    candidate_indices = [int(remaining_indices[pos]) for pos in pool_positions]
    candidate_vars = np.asarray(
        [float(remaining_variances[pos]) for pos in pool_positions],
        dtype=float,
    )
    candidate_scores = np.asarray(
        [float(remaining_scores[pos]) for pos in pool_positions],
        dtype=float,
    )
    candidate_rank = {
        int(remaining_indices[int(pos)]): int(rank)
        for rank, pos in enumerate(order)
    }

    selected = [int(i) for i in selected_context]
    picked: List[int] = []
    picked_diag: List[Dict[str, Any]] = []
    gain_floor = float(max(0.0, novelty_floor))
    power = float(score_power)
    pivot_eps = max(float(jitter), 1.0e-300)
    diag_eps = max(gain_floor, pivot_eps)
    degenerate_candidates = set()
    selected_inv: Optional[np.ndarray] = None
    selected_diag = np.zeros(0, dtype=float)
    k_xs = np.zeros((len(candidate_indices), 0), dtype=float)
    if selected:
        selected_points = [training_atoms[i] for i in selected]
        selected_cov = _posterior_cross_covariances(
            posterior,
            selected_points,
            selected_points,
            chunk_size=variance_chunk_size,
        )
        selected_cov = 0.5 * (selected_cov + selected_cov.T)
        selected_cov = selected_cov + np.eye(selected_cov.shape[0], dtype=float) * float(jitter)
        try:
            selected_inv = np.linalg.inv(selected_cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("D-optimal seed covariance inverse failed") from exc
        selected_diag = np.maximum(np.diag(selected_cov), diag_eps)
        k_xs = _posterior_cross_covariances(
            posterior,
            [training_atoms[i] for i in candidate_indices],
            selected_points,
            chunk_size=variance_chunk_size,
        )

    while candidate_indices and len(picked) < int(n_select):
        if selected_inv is not None and k_xs.shape[1] > 0:
            solved = selected_inv @ k_xs.T
            conditional = candidate_vars - np.einsum("ij,ji->i", k_xs, solved)
            denom = np.sqrt(np.maximum(candidate_vars, diag_eps)[:, None] * selected_diag[None, :])
            max_corr = np.max(np.abs(k_xs) / denom, axis=1)
        else:
            conditional = np.array(candidate_vars, dtype=float)
            max_corr = np.zeros(len(candidate_indices), dtype=float)

        if not np.all(np.isfinite(conditional)):
            raise ValueError("D-optimal conditional variance contains non-finite values")
        if not np.all(np.isfinite(max_corr)):
            raise ValueError("D-optimal correlation diagnostics contain non-finite values")

        raw_scores = np.maximum(candidate_scores, 0.0) ** power
        conditional_floor = np.maximum(conditional, gain_floor)
        usable = conditional > pivot_eps
        for idx, is_usable in zip(candidate_indices, usable):
            if not bool(is_usable):
                degenerate_candidates.add(int(idx))
        gains = np.where(usable, raw_scores * conditional_floor, -np.inf)
        if not np.all(np.isfinite(gains) | np.isneginf(gains)):
            raise ValueError("D-optimal gain contains non-finite values")
        if not np.any(np.isfinite(gains)):
            break

        finite_positions = np.where(np.isfinite(gains))[0]
        pick_order = _quantised_descending_order(
            gains[finite_positions],
            np.asarray(candidate_indices, dtype=int)[finite_positions],
        )
        pick_pos = int(finite_positions[int(pick_order[0])])
        pick_index = int(candidate_indices[pick_pos])
        pick_cov_to_selected = (
            np.array(k_xs[pick_pos, :], dtype=float)
            if k_xs.shape[1] > 0
            else np.zeros(0, dtype=float)
        )
        pick_variance_with_jitter = float(candidate_vars[pick_pos]) + float(jitter)
        picked.append(pick_index)
        selected.append(pick_index)
        picked_diag.append({
            "selection_index": pick_index,
            "selection_origin": "d_optimal",
            "raw_variance": float(candidate_vars[pick_pos]),
            "raw_score": float(candidate_scores[pick_pos]),
            "d_optimal_conditional_variance": float(conditional_floor[pick_pos]),
            "d_optimal_raw_conditional_variance": float(conditional[pick_pos]),
            "d_optimal_gain": float(gains[pick_pos]),
            "d_optimal_prefilter_rank": int(candidate_rank[pick_index]),
            "d_optimal_max_correlation_to_selected": float(max_corr[pick_pos]),
        })
        if selected_inv is None or selected_inv.size == 0:
            selected_inv = np.array([[1.0 / max(pick_variance_with_jitter, pivot_eps)]], dtype=float)
        else:
            inv_b = selected_inv @ pick_cov_to_selected.reshape(-1, 1)
            explained = (pick_cov_to_selected.reshape(1, -1) @ inv_b).item()
            schur = float(pick_variance_with_jitter - float(explained))
            if not np.isfinite(schur) or schur <= pivot_eps:
                degenerate_candidates.add(pick_index)
                picked.pop()
                selected.pop()
                picked_diag.pop()
                del candidate_indices[pick_pos]
                candidate_vars = np.delete(candidate_vars, pick_pos)
                candidate_scores = np.delete(candidate_scores, pick_pos)
                if k_xs.size:
                    k_xs = np.delete(k_xs, pick_pos, axis=0)
                continue
            schur = max(schur, pivot_eps)
            top_left = selected_inv + (inv_b @ inv_b.T) / schur
            top_right = -inv_b / schur
            bottom = np.array([[1.0 / schur]], dtype=float)
            selected_inv = np.block([[top_left, top_right], [top_right.T, bottom]])
        selected_diag = np.append(selected_diag, max(pick_variance_with_jitter, diag_eps))
        new_col = None
        if len(candidate_indices) > 1:
            new_col = _posterior_cross_covariances(
                posterior,
                [training_atoms[i] for i in candidate_indices],
                [training_atoms[pick_index]],
                chunk_size=variance_chunk_size,
            ).reshape(-1, 1)
        del candidate_indices[pick_pos]
        candidate_vars = np.delete(candidate_vars, pick_pos)
        candidate_scores = np.delete(candidate_scores, pick_pos)
        if k_xs.size:
            k_xs = np.delete(k_xs, pick_pos, axis=0)
        if new_col is not None:
            new_col = np.delete(new_col, pick_pos, axis=0)
            k_xs = np.hstack([k_xs, new_col]) if k_xs.size else new_col

    return picked, picked_diag, {
        "prefilter_pool_size": int(n_pool),
        "d_optimal_requested": int(n_select),
        "d_optimal_selected": int(len(picked)),
        "d_optimal_skipped_degenerate": int(len(degenerate_candidates)),
    }


@dataclass(frozen=True)
class SeedSelection:
    """Structured result of `select_seeds`."""

    seeds: List[Atoms]
    indices: List[int]
    bulk_indices: List[int]
    variance_indices: List[int]
    variances: List[float]
    frame_ids: List[Optional[int]] = field(default_factory=list)
    skipped_unknown_provenance: int = 0
    selection_origins: List[str] = field(default_factory=list)
    selection_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.indices)


def select_seeds(
    training_atoms: Sequence[Atoms],
    posterior,
    n_seeds: int,
    *,
    bulk_fraction: float = 0.5,
    rng_seed: int = 0,
    training_frame_ids: Optional[Sequence[Optional[int]]] = None,
    forbidden_frame_ids: FrozenSet[int] = frozenset(),
    variance_chunk_size: Optional[int] = None,
    strategy: str = "hybrid_variance",
    d_optimal_pool_multiplier: int = 8,
    d_optimal_jitter: float = 1.0e-12,
    d_optimal_novelty_floor: float = 1.0e-12,
    d_optimal_score_power: float = 1.0,
    score_transform: Optional[Callable[[int, float], Optional[float]]] = None,
) -> SeedSelection:
    """Return n_seeds seeds from training_atoms.

    With the default ``hybrid_variance`` strategy, the non-random part is the
    highest-variance points by posterior.variance(x). With ``d_optimal``, the
    non-random part is selected greedily by posterior conditional variance
    against the already selected batch seeds.

    Parameters
    ----------
    training_atoms
        Sequence of training :class:Atoms.
    posterior
        Object exposing variance(atoms) -> float. The
        :class:ichor.core.adversarial.posterior.TotalEnergyPosterior already
        satisfies this contract.
    n_seeds
        Total number of seeds to return. If >= number of eligible positions,
        all eligible positions are returned.
    bulk_fraction
        Fraction (in [0, 1]) of n_seeds allocated to the random bulk pick.
        The remainder goes to the variance-weighted pick.
    rng_seed
        Seed for the random bulk selection. Variance ranking is deterministic
        for a given posterior.
    training_frame_ids
        Optional parallel sequence of stable trajectory frame_ids
        (one per training_atoms entry, with None for points without a
        frame_id). Used together with forbidden_frame_ids to filter the
        eligible-position set. When None (the legacy default) no filtering
        is performed and the returned frame_ids are all None.
    forbidden_frame_ids
        Stable trajectory frame_ids that MUST NOT be picked (the union of the
        training-pool provenance frame_ids and the recent-seeds cooldown
        cache). Positions whose training_frame_ids[i] is in this set are
        excluded before sampling.
    variance_chunk_size
        Optional chunk size passed to posterior.variances when the posterior
        supports that keyword.
    strategy
        ``hybrid_variance`` keeps the existing variance-rank path. ``d_optimal``
        replaces only that non-random path with the greedy model-space selector.
    score_transform
        Optional cheap transform for D-optimal ranking/gain scores. It receives
        ``(training_index, posterior_variance)`` and should return a finite,
        non-negative score. ``None``/non-finite values fall back to variance.

    Returns
    -------
    SeedSelection
        seeds in the order: bulk first, then variance. frame_ids is
        parallel to seeds (None entries are preserved when the caller
        passed no frame_ids or when the picked training row has None).
    """
    if not 0.0 <= bulk_fraction <= 1.0:
        raise ValueError(f"bulk_fraction must be in [0, 1]; got {bulk_fraction}")
    strategy = str(strategy)
    if strategy not in VALID_STRATEGIES:
        raise ValueError("seed selection strategy must be one of " + repr(sorted(VALID_STRATEGIES)))
    if int(d_optimal_pool_multiplier) <= 0:
        raise ValueError("d_optimal_pool_multiplier must be > 0")
    if float(d_optimal_jitter) <= 0.0:
        raise ValueError("d_optimal_jitter must be > 0")
    if float(d_optimal_novelty_floor) < 0.0:
        raise ValueError("d_optimal_novelty_floor must be >= 0")
    if float(d_optimal_score_power) < 0.0:
        raise ValueError("d_optimal_score_power must be >= 0")

    n_total = len(training_atoms)
    if n_total <= 0:
        raise ValueError("training_atoms is empty")
    if n_seeds <= 0:
        return SeedSelection([], [], [], [], [], [], 0, [], [], {"strategy": strategy})

    #build the parallel frame_ids vector. If the caller passed None, treat
    #every training row as having no frame_id (and therefore unfilterable).
    if training_frame_ids is None:
        fids: List[Optional[int]] = [None] * n_total
    else:
        if len(training_frame_ids) != n_total:
            raise ValueError(
                "training_frame_ids length "
                + str(len(training_frame_ids))
                + " != training_atoms length " + str(n_total)
            )
        fids = [int(f) if isinstance(f, int) else None for f in training_frame_ids]

    forbidden = frozenset(int(f) for f in forbidden_frame_ids)

    # Filter eligible positions. When the forbidden ledger is active, unknown provenance is
    # excluded instead of being silently eligible: otherwise old pointdirs without frame_id metadata
    # can bypass the anti-repeat guard and re-seed already-forbidden trajectory frames.
    skipped_unknown = sum(1 for fid in fids if fid is None) if forbidden else 0
    eligible: List[int] = [
        i for i in range(n_total)
        if (fids[i] is not None and fids[i] not in forbidden) or (not forbidden and fids[i] is None)
    ]
    n_eligible = len(eligible)
    if n_eligible <= 0:
        #forbidden set ate every training point; return empty selection.
        return SeedSelection(
            [], [], [], [], [], [], skipped_unknown, [], [],
            {
                "strategy": strategy,
                "n_total": int(n_total),
                "n_eligible": 0,
                "skipped_unknown_provenance": int(skipped_unknown),
            },
        )

    if n_seeds >= n_eligible:
        all_idx = list(eligible)
        raw_variances = _posterior_variances(
            posterior,
            [training_atoms[i] for i in all_idx],
            chunk_size=variance_chunk_size,
        )
        variances = [
            float(v)
            for v in _finite_variances(raw_variances, context="seed selection")
        ]
        diagnostics = [
            {
                "selection_index": int(idx),
                "selection_origin": "bulk",
                "raw_variance": float(var),
            }
            for idx, var in zip(all_idx, variances)
        ]
        return SeedSelection(
            seeds=[training_atoms[i] for i in all_idx],
            indices=all_idx,
            bulk_indices=all_idx,
            variance_indices=[],
            variances=variances,
            frame_ids=[fids[i] for i in all_idx],
            skipped_unknown_provenance=skipped_unknown,
            selection_origins=["bulk"] * len(all_idx),
            selection_diagnostics=diagnostics,
            diagnostics={
                "strategy": strategy,
                "n_total": int(n_total),
                "n_eligible": int(n_eligible),
                "n_bulk": int(len(all_idx)),
                "n_ranked": 0,
                "prefilter_pool_size": 0,
                "skipped_unknown_provenance": int(skipped_unknown),
                "d_optimal_bypassed_all_eligible_bulk": bool(strategy == "d_optimal"),
                "ranking_tie_break_policy": "quantised_score_then_index",
                "ranking_score_quantisation_abs": float(RANKING_SCORE_QUANTISATION),
                "n_variance_score_ties_after_quantisation": int(
                    _quantised_tie_count(variances)
                ),
            },
        )

    rng = np.random.default_rng(int(rng_seed))
    n_bulk = int(round(bulk_fraction * n_seeds))
    n_bulk = max(0, min(n_bulk, n_seeds))
    n_variance = n_seeds - n_bulk

    eligible_arr = np.asarray(eligible, dtype=int)
    if n_bulk > 0:
        bulk_pick = rng.choice(len(eligible_arr), size=n_bulk, replace=False)
        bulk_idx = [int(eligible_arr[p]) for p in bulk_pick]
    else:
        bulk_idx = []
    bulk_set = set(bulk_idx)

    remaining = [i for i in eligible if i not in bulk_set]
    # batched scan when the posterior supports it (the real GP). avoids a
    # python variance() call per pool frame on the login node.
    remaining_vars = np.asarray(
        _posterior_variances(
            posterior,
            [training_atoms[i] for i in remaining],
            chunk_size=variance_chunk_size,
        ),
        dtype=float,
    )
    remaining_vars = _finite_variances(remaining_vars, context="seed ranking")
    remaining_scores = np.array(remaining_vars, dtype=float)
    if score_transform is not None:
        transformed = []
        for idx, var in zip(remaining, remaining_vars):
            value = score_transform(int(idx), float(var))
            try:
                score = float(value) if value is not None else float(var)
            except (TypeError, ValueError):
                score = float(var)
            if not np.isfinite(score) or score < 0.0:
                score = float(var)
            transformed.append(score)
        remaining_scores = _finite_variances(transformed, context="seed ranking score")
    selection_diag_by_index: Dict[int, Dict[str, Any]] = {}
    if remaining:
        order = _quantised_descending_order(remaining_vars, remaining)
        if strategy == "hybrid_variance":
            variance_idx = [
                remaining[int(order[k])]
                for k in range(min(n_variance, len(remaining)))
            ]
            for rank, pos in enumerate(order):
                idx = int(remaining[int(pos)])
                if idx in variance_idx:
                    selection_diag_by_index[idx] = {
                        "selection_index": idx,
                        "selection_origin": "variance",
                        "raw_variance": float(remaining_vars[int(pos)]),
                        "variance_rank": int(rank),
                    }
        else:
            variance_idx, dopt_diags, dopt_summary = _d_optimal_select(
                training_atoms=training_atoms,
                posterior=posterior,
                remaining_indices=remaining,
                remaining_variances=remaining_vars,
                remaining_scores=remaining_scores,
                selected_context=bulk_idx,
                n_select=min(n_variance, len(remaining)),
                pool_multiplier=int(d_optimal_pool_multiplier),
                jitter=float(d_optimal_jitter),
                novelty_floor=float(d_optimal_novelty_floor),
                score_power=float(d_optimal_score_power),
                variance_chunk_size=variance_chunk_size,
            )
            selection_diag_by_index.update({int(d["selection_index"]): dict(d) for d in dopt_diags})
    else:
        variance_idx = []
        dopt_summary = {
            "prefilter_pool_size": 0,
            "d_optimal_requested": int(n_variance),
            "d_optimal_selected": 0,
        }

    all_idx = bulk_idx + variance_idx
    variances = [
        float(v)
        for v in _finite_variances(
            _posterior_variances(
                posterior,
                [training_atoms[i] for i in all_idx],
                chunk_size=variance_chunk_size,
            ),
            context="seed selection",
        )
    ]
    variance_by_idx = {int(idx): float(var) for idx, var in zip(all_idx, variances)}
    selection_origins = []
    selection_diagnostics: List[Dict[str, Any]] = []
    for idx in all_idx:
        if int(idx) in bulk_set:
            origin = "bulk"
            diag = {
                "selection_index": int(idx),
                "selection_origin": origin,
                "raw_variance": float(variance_by_idx[int(idx)]),
            }
        else:
            origin = "d_optimal" if strategy == "d_optimal" else "variance"
            diag = dict(selection_diag_by_index.get(int(idx), {}))
            diag.setdefault("selection_index", int(idx))
            diag.setdefault("selection_origin", origin)
            diag.setdefault("raw_variance", float(variance_by_idx[int(idx)]))
        selection_origins.append(origin)
        selection_diagnostics.append(diag)

    summary = {
        "strategy": strategy,
        "n_total": int(n_total),
        "n_eligible": int(n_eligible),
        "n_bulk": int(len(bulk_idx)),
        "n_ranked": int(len(variance_idx)),
        "skipped_unknown_provenance": int(skipped_unknown),
        "ranking_tie_break_policy": "quantised_score_then_index",
        "ranking_score_quantisation_abs": float(RANKING_SCORE_QUANTISATION),
        "n_variance_score_ties_after_quantisation": int(
            _quantised_tie_count(remaining_scores if remaining else [])
        ),
    }
    if strategy == "d_optimal":
        summary.update(dopt_summary)
    else:
        summary["prefilter_pool_size"] = int(len(remaining))

    return SeedSelection(
        seeds=[training_atoms[i] for i in all_idx],
        indices=all_idx,
        bulk_indices=bulk_idx,
        variance_indices=variance_idx,
        variances=variances,
        frame_ids=[fids[i] for i in all_idx],
        skipped_unknown_provenance=skipped_unknown,
        selection_origins=selection_origins,
        selection_diagnostics=selection_diagnostics,
        diagnostics=summary,
    )
