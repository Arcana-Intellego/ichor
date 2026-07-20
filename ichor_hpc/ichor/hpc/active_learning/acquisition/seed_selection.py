"""Seed selection for the adversarial attack phase.

By default, part of the batch is sampled uniformly at random from the eligible
trajectory pool and the remainder is chosen by top posterior variance. Users
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
from scipy.linalg import solve_triangular

from ichor.core.atoms import Atoms

from ..selection_origins import SEED_SELECTION_ORIGINS


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


ProgressCallback = Callable[[str, Dict[str, Any]], None]


def _emit_progress(
    callback: Optional[ProgressCallback], stage: str, **payload: Any
) -> None:
    if callback is not None:
        callback(str(stage), dict(payload))


def _posterior_variances(
    posterior,
    points,
    *,
    chunk_size: Optional[int],
    indices: Optional[Sequence[int]] = None,
):
    if indices is not None and hasattr(posterior, "variances_by_index"):
        return posterior.variances_by_index(indices)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
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
    left_indices: Optional[Sequence[int]] = None,
    right_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Return covariance(left, right) without asking for a full pool matrix."""
    n_left = len(left_points)
    n_right = len(right_points)
    if n_left == 0 or n_right == 0:
        return np.zeros((n_left, n_right), dtype=float)
    if (
        left_indices is not None
        and right_indices is not None
        and hasattr(posterior, "cross_covariances_by_index")
    ):
        return _finite_matrix(
            posterior.cross_covariances_by_index(left_indices, right_indices),
            context="D-optimal prepared posterior cross-covariance",
        )
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


def _d_optimal_select_legacy(
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
    degenerate_policy: str,
    variance_chunk_size: Optional[int],
    progress_callback: Optional[ProgressCallback] = None,
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
    _emit_progress(
        progress_callback,
        "d_optimal",
        completed=0,
        total=int(n_select),
        shortlist_size=int(n_pool),
        remaining_candidates=int(n_pool),
    )

    if degenerate_policy not in {"fail", "score_backfill"}:
        raise ValueError("unknown D-optimal degenerate policy: " + repr(degenerate_policy))
    selected = [int(i) for i in selected_context]
    picked: List[int] = []
    picked_diag: List[Dict[str, Any]] = []
    gain_floor = float(max(0.0, novelty_floor))
    power = float(score_power)
    variance_scale = max(
        float(np.max(candidate_vars)) if candidate_vars.size else 0.0,
        1.0,
    )
    pivot_eps = max(
        10.0 * float(jitter),
        np.finfo(float).eps * variance_scale * 64.0,
    )
    diagnostic_floor = max(gain_floor, pivot_eps)
    degenerate_candidates: set[int] = set()
    last_conditional: Dict[int, float] = {}
    last_correlation: Dict[int, float] = {}
    selected_cholesky: Optional[np.ndarray] = None
    selected_diag = np.zeros(0, dtype=float)
    k_xs = np.zeros((len(candidate_indices), 0), dtype=float)
    if selected:
        selected_points = [training_atoms[i] for i in selected]
        selected_cov = _posterior_cross_covariances(
            posterior,
            selected_points,
            selected_points,
            chunk_size=variance_chunk_size,
            left_indices=selected,
            right_indices=selected,
        )
        selected_cov = 0.5 * (selected_cov + selected_cov.T)
        selected_cov = selected_cov + np.eye(selected_cov.shape[0], dtype=float) * float(jitter)
        try:
            selected_cholesky = np.linalg.cholesky(selected_cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("D-optimal seed covariance Cholesky factor failed") from exc
        selected_diag = np.maximum(np.diag(selected_cov), diagnostic_floor)
        k_xs = _posterior_cross_covariances(
            posterior,
            [training_atoms[i] for i in candidate_indices],
            selected_points,
            chunk_size=variance_chunk_size,
            left_indices=candidate_indices,
            right_indices=selected,
        )

    while candidate_indices and len(picked) < int(n_select):
        if selected_cholesky is not None and k_xs.shape[1] > 0:
            projected = np.linalg.solve(selected_cholesky, k_xs.T)
            conditional = candidate_vars - np.sum(np.square(projected), axis=0)
            denom = np.sqrt(
                np.maximum(candidate_vars, diagnostic_floor)[:, None]
                * selected_diag[None, :]
            )
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
        for position, (idx, is_usable) in enumerate(zip(candidate_indices, usable)):
            last_conditional[int(idx)] = float(conditional[position])
            last_correlation[int(idx)] = float(max_corr[position])
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
        if selected_cholesky is None or selected_cholesky.size == 0:
            schur = pick_variance_with_jitter
            if not np.isfinite(schur) or schur <= pivot_eps:
                degenerate_candidates.add(pick_index)
                del candidate_indices[pick_pos]
                candidate_vars = np.delete(candidate_vars, pick_pos)
                candidate_scores = np.delete(candidate_scores, pick_pos)
                continue
            selected_cholesky = np.array([[float(np.sqrt(schur))]], dtype=float)
        else:
            projected_pick = np.linalg.solve(
                selected_cholesky,
                pick_cov_to_selected,
            )
            schur = float(
                pick_variance_with_jitter
                - float(np.dot(projected_pick, projected_pick))
            )
            if not np.isfinite(schur) or schur <= pivot_eps:
                degenerate_candidates.add(pick_index)
                del candidate_indices[pick_pos]
                candidate_vars = np.delete(candidate_vars, pick_pos)
                candidate_scores = np.delete(candidate_scores, pick_pos)
                if k_xs.size:
                    k_xs = np.delete(k_xs, pick_pos, axis=0)
                continue
            old_size = int(selected_cholesky.shape[0])
            expanded = np.zeros((old_size + 1, old_size + 1), dtype=float)
            expanded[:old_size, :old_size] = selected_cholesky
            expanded[old_size, :old_size] = projected_pick
            expanded[old_size, old_size] = float(np.sqrt(schur))
            selected_cholesky = expanded
        picked.append(pick_index)
        selected.append(pick_index)
        picked_diag.append({
            "selection_index": pick_index,
            "selection_origin": "d_optimal",
            "raw_variance": float(candidate_vars[pick_pos]),
            "raw_score": float(candidate_scores[pick_pos]),
            "d_optimal_conditional_variance": float(max(conditional[pick_pos], 0.0)),
            "d_optimal_raw_conditional_variance": float(conditional[pick_pos]),
            "d_optimal_gain": float(gains[pick_pos]),
            "d_optimal_prefilter_rank": int(candidate_rank[pick_index]),
            "d_optimal_max_correlation_to_selected": float(max_corr[pick_pos]),
        })
        _emit_progress(
            progress_callback,
            "d_optimal",
            completed=int(len(picked)),
            total=int(n_select),
            shortlist_size=int(n_pool),
            remaining_candidates=int(len(candidate_indices) - 1),
        )
        selected_diag = np.append(
            selected_diag,
            max(pick_variance_with_jitter, diagnostic_floor),
        )
        new_col = None
        if len(candidate_indices) > 1:
            new_col = _posterior_cross_covariances(
                posterior,
                [training_atoms[i] for i in candidate_indices],
                [training_atoms[pick_index]],
                chunk_size=variance_chunk_size,
                left_indices=candidate_indices,
                right_indices=[pick_index],
            ).reshape(-1, 1)
        del candidate_indices[pick_pos]
        candidate_vars = np.delete(candidate_vars, pick_pos)
        candidate_scores = np.delete(candidate_scores, pick_pos)
        if k_xs.size:
            k_xs = np.delete(k_xs, pick_pos, axis=0)
        if new_col is not None:
            new_col = np.delete(new_col, pick_pos, axis=0)
            k_xs = np.hstack([k_xs, new_col]) if k_xs.size else new_col

    n_d_optimal = len(picked)
    n_backfill = max(0, int(n_select) - n_d_optimal)
    if n_backfill:
        if degenerate_policy == "fail":
            raise ValueError(
                "D-optimal model-space degeneracy selected "
                + str(n_d_optimal)
                + " of "
                + str(int(n_select))
                + " requested seeds"
            )
        remaining_by_score = _quantised_descending_order(
            remaining_variances,
            remaining_indices,
        )
        picked_set = set(picked)
        backfill_positions = [
            int(position)
            for position in remaining_by_score
            if int(remaining_indices[int(position)]) not in picked_set
        ][:n_backfill]
        if len(backfill_positions) != n_backfill:
            raise ValueError("D-optimal score backfill could not satisfy requested seed count")
        for position in backfill_positions:
            index = int(remaining_indices[position])
            picked.append(index)
            picked_diag.append({
                "selection_index": index,
                "selection_origin": "d_optimal_backfill",
                "raw_variance": float(remaining_variances[position]),
                "raw_score": float(remaining_scores[position]),
                "d_optimal_conditional_variance": (
                    max(0.0, float(last_conditional[index]))
                    if index in last_conditional
                    else None
                ),
                "d_optimal_raw_conditional_variance": last_conditional.get(index),
                "d_optimal_gain": None,
                "d_optimal_prefilter_rank": int(candidate_rank.get(index, position)),
                "d_optimal_max_correlation_to_selected": last_correlation.get(index),
                "d_optimal_backfill_reason": "model_space_degeneracy",
                "d_optimal_backfill_rank_source": "raw_posterior_variance",
            })

    return picked, picked_diag, {
        "prefilter_pool_size": int(n_pool),
        "d_optimal_requested": int(n_select),
        "d_optimal_selected": int(n_d_optimal),
        "d_optimal_backfilled": int(n_backfill),
        "d_optimal_total_selected": int(len(picked)),
        "d_optimal_degenerate_policy": str(degenerate_policy),
        "d_optimal_pivot_floor": float(pivot_eps),
        "d_optimal_skipped_degenerate": int(len(degenerate_candidates)),
    }


def _prepared_triangular_solve(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lower = _finite_matrix(factor, context="D-optimal Cholesky factor")
    values = np.asarray(rhs, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != lower.shape[0]:
        raise ValueError("D-optimal triangular solve shape mismatch")
    if lower.shape[0] != lower.shape[1] or np.any(np.diag(lower) <= 0.0):
        raise ValueError("D-optimal Cholesky factor is not positive triangular")
    if not np.all(np.isfinite(values)):
        raise ValueError("D-optimal triangular right-hand side is non-finite")
    solved = solve_triangular(
        lower,
        values,
        lower=True,
        check_finite=False,
        overwrite_b=False,
    )
    if not np.all(np.isfinite(solved)):
        raise ValueError("D-optimal triangular solution is non-finite")
    return np.asarray(solved, dtype=float)


def _d_optimal_select_prepared(
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
    degenerate_policy: str,
    variance_chunk_size: Optional[int],
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[List[int], List[Dict[str, Any]], Dict[str, Any]]:
    """Prepared greedy D-optimal selection with rank-one residual updates."""
    if n_select <= 0 or not remaining_indices:
        return [], [], {
            "prefilter_pool_size": 0,
            "d_optimal_requested": int(max(0, n_select)),
            "d_optimal_selected": 0,
            "d_optimal_prepared": True,
        }
    n_pool = min(
        len(remaining_indices),
        max(int(n_select), int(n_select) * int(pool_multiplier)),
    )
    order = _quantised_descending_order(remaining_scores, remaining_indices)
    pool_positions = [int(position) for position in order[:n_pool]]
    candidate_indices = [int(remaining_indices[position]) for position in pool_positions]
    candidate_vars = np.asarray(
        [float(remaining_variances[position]) for position in pool_positions],
        dtype=float,
    )
    candidate_scores = np.asarray(
        [float(remaining_scores[position]) for position in pool_positions],
        dtype=float,
    )
    candidate_rank = {
        int(remaining_indices[int(position)]): int(rank)
        for rank, position in enumerate(order)
    }
    selected = [int(index) for index in selected_context]
    prepared_ids = list(dict.fromkeys(selected + candidate_indices))
    prepared = posterior.prepare_indexed_batch(prepared_ids)
    if not hasattr(prepared, "cross_covariances_by_index"):
        raise ValueError("prepared D-optimal posterior lacks indexed covariance")

    if degenerate_policy not in {"fail", "score_backfill"}:
        raise ValueError("unknown D-optimal degenerate policy: " + repr(degenerate_policy))
    picked: List[int] = []
    picked_diag: List[Dict[str, Any]] = []
    gain_floor = float(max(0.0, novelty_floor))
    power = float(score_power)
    variance_scale = max(
        float(np.max(candidate_vars)) if candidate_vars.size else 0.0,
        1.0,
    )
    pivot_eps = max(
        10.0 * float(jitter),
        np.finfo(float).eps * variance_scale * 64.0,
    )
    diagnostic_floor = max(gain_floor, pivot_eps)
    degenerate_candidates: set[int] = set()
    last_conditional: Dict[int, float] = {}
    last_correlation: Dict[int, float] = {}
    selected_cholesky: Optional[np.ndarray] = None
    selected_diag = np.zeros(0, dtype=float)
    k_xs = np.zeros((len(candidate_indices), 0), dtype=float)
    projected = np.zeros((len(candidate_indices), 0), dtype=float)
    ambiguity_rechecks = 0

    if selected:
        selected_cov = _finite_matrix(
            prepared.cross_covariances_by_index(selected, selected),
            context="prepared selected covariance",
        )
        selected_cov = 0.5 * (selected_cov + selected_cov.T)
        selected_cov = selected_cov + np.eye(len(selected), dtype=float) * float(jitter)
        try:
            selected_cholesky = np.linalg.cholesky(selected_cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("D-optimal seed covariance Cholesky factor failed") from exc
        selected_diag = np.maximum(np.diag(selected_cov), diagnostic_floor)
        k_xs = _finite_matrix(
            prepared.cross_covariances_by_index(candidate_indices, selected),
            context="prepared candidate-selected covariance",
        )
        projected = _prepared_triangular_solve(selected_cholesky, k_xs.T).T

    _emit_progress(
        progress_callback,
        "d_optimal",
        completed=0,
        total=int(n_select),
        shortlist_size=int(n_pool),
        remaining_candidates=int(n_pool),
        prepared=True,
    )

    while candidate_indices and len(picked) < int(n_select):
        conditional = candidate_vars - np.sum(projected * projected, axis=1)
        if k_xs.shape[1] > 0:
            denominator = np.sqrt(
                np.maximum(candidate_vars, diagnostic_floor)[:, None]
                * selected_diag[None, :]
            )
            max_corr = np.max(np.abs(k_xs) / denominator, axis=1)
        else:
            max_corr = np.zeros(len(candidate_indices), dtype=float)
        if not np.all(np.isfinite(conditional)):
            raise ValueError("D-optimal conditional variance contains non-finite values")
        if not np.all(np.isfinite(max_corr)):
            raise ValueError("D-optimal correlation diagnostics contain non-finite values")

        raw_scores = np.maximum(candidate_scores, 0.0) ** power
        usable = conditional > pivot_eps
        for position, (index, is_usable) in enumerate(zip(candidate_indices, usable)):
            last_conditional[int(index)] = float(conditional[position])
            last_correlation[int(index)] = float(max_corr[position])
            if not bool(is_usable):
                degenerate_candidates.add(int(index))
        gains = np.where(
            usable,
            raw_scores * np.maximum(conditional, gain_floor),
            -np.inf,
        )
        if not np.all(np.isfinite(gains) | np.isneginf(gains)):
            raise ValueError("D-optimal gain contains non-finite values")
        finite_positions = np.where(np.isfinite(gains))[0]
        if finite_positions.size == 0:
            break
        provisional_order = _quantised_descending_order(
            gains[finite_positions],
            np.asarray(candidate_indices, dtype=int)[finite_positions],
        )
        provisional_position = int(finite_positions[int(provisional_order[0])])

        if selected_cholesky is not None and selected_cholesky.size:
            rounding = (
                np.finfo(float).eps
                * (64.0 + 8.0 * float(selected_cholesky.shape[0]))
                * np.maximum(
                    1.0,
                    np.abs(candidate_vars) + np.sum(projected * projected, axis=1),
                )
                * np.maximum(raw_scores, 1.0)
            )
            top_gain = float(gains[provisional_position])
            contenders = np.asarray(
                [
                    int(position)
                    for position in finite_positions
                    if top_gain - float(gains[int(position)])
                    <= float(RANKING_SCORE_QUANTISATION)
                    + float(rounding[provisional_position])
                    + float(rounding[int(position)])
                ],
                dtype=int,
            )
            if contenders.size > 1:
                direct = _prepared_triangular_solve(
                    selected_cholesky,
                    k_xs[contenders, :].T,
                ).T
                direct_conditional = (
                    candidate_vars[contenders]
                    - np.sum(direct * direct, axis=1)
                )
                projected[contenders, :] = direct
                direct_usable = direct_conditional > pivot_eps
                direct_gains = np.where(
                    direct_usable,
                    raw_scores[contenders]
                    * np.maximum(direct_conditional, gain_floor),
                    -np.inf,
                )
                conditional[contenders] = direct_conditional
                gains[contenders] = direct_gains
                direct_finite = np.where(np.isfinite(direct_gains))[0]
                if direct_finite.size:
                    direct_order = _quantised_descending_order(
                        direct_gains[direct_finite],
                        np.asarray(candidate_indices, dtype=int)[
                            contenders[direct_finite]
                        ],
                    )
                    provisional_position = int(
                        contenders[int(direct_finite[int(direct_order[0])])]
                    )
                ambiguity_rechecks += int(contenders.size)

        pick_pos = int(provisional_position)
        pick_index = int(candidate_indices[pick_pos])
        projected_pick = np.asarray(projected[pick_pos, :], dtype=float)
        pick_variance_with_jitter = float(candidate_vars[pick_pos]) + float(jitter)
        schur = pick_variance_with_jitter - float(
            np.dot(projected_pick, projected_pick)
        )
        if not np.isfinite(schur) or schur <= pivot_eps:
            degenerate_candidates.add(pick_index)
            del candidate_indices[pick_pos]
            candidate_vars = np.delete(candidate_vars, pick_pos)
            candidate_scores = np.delete(candidate_scores, pick_pos)
            projected = np.delete(projected, pick_pos, axis=0)
            k_xs = np.delete(k_xs, pick_pos, axis=0)
            continue

        old_size = int(selected_cholesky.shape[0]) if selected_cholesky is not None else 0
        expanded = np.zeros((old_size + 1, old_size + 1), dtype=float)
        if old_size:
            expanded[:old_size, :old_size] = selected_cholesky
            expanded[old_size, :old_size] = projected_pick
        expanded[old_size, old_size] = float(np.sqrt(schur))
        selected_cholesky = expanded

        picked.append(pick_index)
        selected.append(pick_index)
        picked_diag.append(
            {
                "selection_index": pick_index,
                "selection_origin": "d_optimal",
                "raw_variance": float(candidate_vars[pick_pos]),
                "raw_score": float(candidate_scores[pick_pos]),
                "d_optimal_conditional_variance": float(
                    max(conditional[pick_pos], 0.0)
                ),
                "d_optimal_raw_conditional_variance": float(conditional[pick_pos]),
                "d_optimal_gain": float(gains[pick_pos]),
                "d_optimal_prefilter_rank": int(candidate_rank[pick_index]),
                "d_optimal_max_correlation_to_selected": float(max_corr[pick_pos]),
            }
        )
        selected_diag = np.append(
            selected_diag, max(pick_variance_with_jitter, diagnostic_floor)
        )

        new_raw_column = _finite_matrix(
            prepared.cross_covariances_by_index(candidate_indices, [pick_index]),
            context="prepared D-optimal covariance column",
        ).reshape(-1)
        diagonal_tolerance = max(
            float(RANKING_SCORE_QUANTISATION),
            np.finfo(float).eps
            * max(1, len(selected))
            * max(1.0, abs(float(candidate_vars[pick_pos])))
            * 2048.0,
        )
        if abs(
            float(new_raw_column[pick_pos]) - float(candidate_vars[pick_pos])
        ) > diagonal_tolerance:
            raise ValueError(
                "prepared D-optimal covariance diagonal is inconsistent with "
                "the cached posterior variance"
            )
        residual_column = new_raw_column - projected @ projected_pick
        new_projected_column = residual_column / float(np.sqrt(schur))
        if not np.all(np.isfinite(new_projected_column)):
            raise ValueError("prepared D-optimal rank-one update is non-finite")

        del candidate_indices[pick_pos]
        candidate_vars = np.delete(candidate_vars, pick_pos)
        candidate_scores = np.delete(candidate_scores, pick_pos)
        projected = np.delete(projected, pick_pos, axis=0)
        new_projected_column = np.delete(new_projected_column, pick_pos)
        projected = np.column_stack((projected, new_projected_column))
        updated_conditional = candidate_vars - np.sum(projected * projected, axis=1)
        residual_tolerance = max(
            float(pivot_eps) * 8.0,
            np.finfo(float).eps
            * max(1, int(projected.shape[1]))
            * variance_scale
            * 2048.0,
        )
        if np.any(updated_conditional < -residual_tolerance):
            raise ValueError(
                "prepared D-optimal rank-one residual violated covariance bounds"
            )
        k_xs = np.delete(k_xs, pick_pos, axis=0)
        new_raw_column = np.delete(new_raw_column, pick_pos)
        k_xs = np.column_stack((k_xs, new_raw_column))
        _emit_progress(
            progress_callback,
            "d_optimal",
            completed=int(len(picked)),
            total=int(n_select),
            shortlist_size=int(n_pool),
            remaining_candidates=int(len(candidate_indices)),
            prepared=True,
        )

    n_d_optimal = len(picked)
    n_backfill = max(0, int(n_select) - n_d_optimal)
    if n_backfill:
        if degenerate_policy == "fail":
            raise ValueError(
                "D-optimal model-space degeneracy selected "
                + str(n_d_optimal)
                + " of "
                + str(int(n_select))
                + " requested seeds"
            )
        remaining_by_score = _quantised_descending_order(
            remaining_variances, remaining_indices
        )
        picked_set = set(picked)
        backfill_positions = [
            int(position)
            for position in remaining_by_score
            if int(remaining_indices[int(position)]) not in picked_set
        ][:n_backfill]
        if len(backfill_positions) != n_backfill:
            raise ValueError("D-optimal score backfill could not satisfy requested seed count")
        for position in backfill_positions:
            index = int(remaining_indices[position])
            picked.append(index)
            picked_diag.append(
                {
                    "selection_index": index,
                    "selection_origin": "d_optimal_backfill",
                    "raw_variance": float(remaining_variances[position]),
                    "raw_score": float(remaining_scores[position]),
                    "d_optimal_conditional_variance": (
                        max(0.0, float(last_conditional[index]))
                        if index in last_conditional
                        else None
                    ),
                    "d_optimal_raw_conditional_variance": last_conditional.get(index),
                    "d_optimal_gain": None,
                    "d_optimal_prefilter_rank": int(
                        candidate_rank.get(index, position)
                    ),
                    "d_optimal_max_correlation_to_selected": last_correlation.get(
                        index
                    ),
                    "d_optimal_backfill_reason": "model_space_degeneracy",
                    "d_optimal_backfill_rank_source": "raw_posterior_variance",
                }
            )

    return picked, picked_diag, {
        "prefilter_pool_size": int(n_pool),
        "d_optimal_requested": int(n_select),
        "d_optimal_selected": int(n_d_optimal),
        "d_optimal_backfilled": int(n_backfill),
        "d_optimal_total_selected": int(len(picked)),
        "d_optimal_degenerate_policy": str(degenerate_policy),
        "d_optimal_pivot_floor": float(pivot_eps),
        "d_optimal_skipped_degenerate": int(len(degenerate_candidates)),
        "d_optimal_prepared": True,
        "d_optimal_ambiguity_rechecks": int(ambiguity_rechecks),
    }


def _d_optimal_select(**kwargs):
    posterior = kwargs.get("posterior")
    if hasattr(posterior, "prepare_indexed_batch"):
        try:
            return _d_optimal_select_prepared(**kwargs)
        except (
            ArithmeticError,
            FloatingPointError,
            np.linalg.LinAlgError,
            ValueError,
        ) as exc:
            if not bool(getattr(posterior, "allow_prepared_legacy_fallback", False)):
                raise
            if not bool(getattr(posterior, "prepared_batch_available", False)):
                raise
            picked, rows, diagnostics = _d_optimal_select_legacy(**kwargs)
            diagnostics = {
                **dict(diagnostics),
                "d_optimal_prepared_fallback": True,
                "d_optimal_prepared_fallback_reason": (
                    type(exc).__name__ + ": " + str(exc)[:240]
                ),
            }
            return picked, rows, diagnostics
    return _d_optimal_select_legacy(**kwargs)


@dataclass(frozen=True)
class SeedSelection:
    """Structured result of `select_seeds`."""

    seeds: List[Atoms]
    indices: List[int]
    bulk_indices: List[int]
    variance_indices: List[int]
    variances: List[Optional[float]]
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
    d_optimal_degenerate_policy: str = "score_backfill",
    score_transform: Optional[Callable[..., Optional[float]]] = None,
    progress_callback: Optional[ProgressCallback] = None,
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
        ``(training_index, posterior_variance, population_variance_scale)``
        and should return a finite, non-negative score. Two-argument legacy
        callables remain accepted. ``None``/non-finite values fall back to
        variance.

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
    if str(d_optimal_degenerate_policy) not in {"fail", "score_backfill"}:
        raise ValueError(
            "d_optimal_degenerate_policy must be 'fail' or 'score_backfill'"
        )

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
    _emit_progress(
        progress_callback,
        "filtering",
        completed=0,
        total=int(n_total),
        forbidden=int(len(forbidden)),
    )

    # Filter eligible positions. When the forbidden ledger is active, unknown provenance is
    # excluded instead of being silently eligible: otherwise old pointdirs without frame_id metadata
    # can bypass the anti-repeat guard and re-seed already-forbidden trajectory frames.
    skipped_unknown = sum(1 for fid in fids if fid is None) if forbidden else 0
    eligible: List[int] = [
        i for i in range(n_total)
        if (fids[i] is not None and fids[i] not in forbidden) or (not forbidden and fids[i] is None)
    ]
    n_eligible = len(eligible)
    _emit_progress(
        progress_callback,
        "filtering",
        completed=int(n_total),
        total=int(n_total),
        eligible=int(n_eligible),
        forbidden=int(len(forbidden)),
    )
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
        variances: List[Optional[float]] = [None] * len(all_idx)
        diagnostics = [
            {
                "selection_index": int(idx),
                "selection_origin": "bulk",
                "raw_variance": None,
                "variance_not_evaluated_reason": "all_eligible_selected_as_bulk",
            }
            for idx in all_idx
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
                "n_variance_score_ties_after_quantisation": 0,
                "posterior_variance_evaluations": 0,
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
    _emit_progress(
        progress_callback,
        "random",
        completed=int(len(bulk_idx)),
        total=int(n_bulk),
        eligible=int(n_eligible),
    )

    remaining = [i for i in eligible if i not in bulk_set]
    if n_variance == 0:
        diagnostics = [
            {
                "selection_index": int(idx),
                "selection_origin": "bulk",
                "raw_variance": None,
                "variance_not_evaluated_reason": "all_bulk_policy",
            }
            for idx in bulk_idx
        ]
        return SeedSelection(
            seeds=[training_atoms[i] for i in bulk_idx],
            indices=bulk_idx,
            bulk_indices=bulk_idx,
            variance_indices=[],
            variances=[None] * len(bulk_idx),
            frame_ids=[fids[i] for i in bulk_idx],
            skipped_unknown_provenance=skipped_unknown,
            selection_origins=["bulk"] * len(bulk_idx),
            selection_diagnostics=diagnostics,
            diagnostics={
                "strategy": strategy,
                "n_total": int(n_total),
                "n_eligible": int(n_eligible),
                "n_bulk": int(len(bulk_idx)),
                "n_ranked": 0,
                "prefilter_pool_size": 0,
                "skipped_unknown_provenance": int(skipped_unknown),
                "d_optimal_bypassed_all_bulk": bool(strategy == "d_optimal"),
                "ranking_tie_break_policy": "quantised_score_then_index",
                "ranking_score_quantisation_abs": float(
                    RANKING_SCORE_QUANTISATION
                ),
                "n_variance_score_ties_after_quantisation": 0,
                "posterior_variance_evaluations": 0,
            },
        )
    # batched scan when the posterior supports it (the real GP). avoids a
    # python variance() call per pool frame on the login node.
    variance_progress_total = int(
        getattr(posterior, "variance_population_size", len(remaining))
    )
    _emit_progress(
        progress_callback,
        "variance",
        completed=0,
        total=variance_progress_total,
    )
    remaining_vars = np.asarray(
        _posterior_variances(
            posterior,
            (
                []
                if hasattr(posterior, "variances_by_index")
                else [training_atoms[i] for i in remaining]
            ),
            chunk_size=variance_chunk_size,
            indices=remaining,
        ),
        dtype=float,
    )
    remaining_vars = _finite_variances(remaining_vars, context="seed ranking")
    _emit_progress(
        progress_callback,
        "variance",
        completed=variance_progress_total,
        total=variance_progress_total,
    )
    remaining_scores = np.array(remaining_vars, dtype=float)
    if score_transform is not None:
        positive_variances = remaining_vars[remaining_vars > 0.0]
        population_scale = (
            float(np.median(positive_variances))
            if positive_variances.size
            else 1.0
        )
        transformed = []
        for idx, var in zip(remaining, remaining_vars):
            try:
                value = score_transform(
                    int(idx),
                    float(var),
                    population_scale,
                )
            except TypeError as exc:
                if "positional" not in str(exc) and "argument" not in str(exc):
                    raise
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
                degenerate_policy=str(d_optimal_degenerate_policy),
                variance_chunk_size=variance_chunk_size,
                progress_callback=progress_callback,
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
    remaining_variance_by_index = {
        int(index): float(value)
        for index, value in zip(remaining, remaining_vars)
    }
    bulk_variances = _finite_variances(
        _posterior_variances(
            posterior,
            (
                []
                if hasattr(posterior, "variances_by_index")
                else [training_atoms[index] for index in bulk_idx]
            ),
            chunk_size=variance_chunk_size,
            indices=bulk_idx,
        ),
        context="bulk seed selection",
    )
    variance_by_idx = {
        int(index): float(value)
        for index, value in zip(bulk_idx, bulk_variances)
    }
    variance_by_idx.update(
        {
            int(index): float(remaining_variance_by_index[int(index)])
            for index in variance_idx
        }
    )
    variances = [float(variance_by_idx[int(index)]) for index in all_idx]
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
            diag = dict(selection_diag_by_index.get(int(idx), {}))
            origin = (
                str(diag.get("selection_origin") or "d_optimal")
                if strategy == "d_optimal"
                else "variance"
            )
            if origin not in SEED_SELECTION_ORIGINS:
                raise ValueError("unknown seed-selection origin: " + repr(origin))
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
