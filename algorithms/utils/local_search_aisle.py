"""Local-search operators over a binary aisle mask.

Used by the memetic GA to refine the neighborhood of strong heuristic seeds
(`AisleFirstHeuristic`, `SimpleHeuristic`) and the GA's own incumbents.

The chromosome is a 1D binary numpy array of length `n_aisles`. Operators
generate candidate masks ranked by a cheap heuristic so callers can cap the
neighborhood size (`neighbor_cap`) for scalability on large instances.
"""

from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np

from algorithms.utils.similarity import similarity

VALID_OPERATORS = {"remove", "add", "swap"}
VALID_STRATEGIES = {"first_improvement", "best_improvement"}


def _useful_score(aisle: dict[int, int], gap: dict[int, int]) -> int:
    return sum(min(qty, gap.get(item, 0)) for item, qty in aisle.items())


def _aisle_units(aisle: dict[int, int]) -> int:
    return sum(aisle.values())


def remove_neighbors(
    mask: np.ndarray,
    aisles: list[dict[int, int]],
    served_demand: dict[int, int],
    cap: int,
) -> Iterable[np.ndarray]:
    """Generate masks with one active aisle removed, prioritizing low-utility ones.

    Aisles that contribute little to the served demand are tried first — removing
    them most likely shrinks the denominator without losing orders.
    """
    active = np.flatnonzero(mask).tolist()
    if not active:
        return []
    scored = sorted(active, key=lambda i: _useful_score(aisles[i], served_demand))
    out = []
    for idx in scored[:cap]:
        cand = mask.copy()
        cand[idx] = 0
        out.append(cand)
    return out


def add_neighbors(
    mask: np.ndarray,
    aisles: list[dict[int, int]],
    unmet_demand: dict[int, int],
    cap: int,
) -> Iterable[np.ndarray]:
    """Generate masks with one inactive aisle added, prioritizing high coverage of unmet demand.

    Adding an aisle raises the denominator, so it only pays off when it unlocks
    enough new orders. Aisles with high `useful` against unmet demand go first.
    """
    inactive = np.flatnonzero(mask == 0).tolist()
    if not inactive:
        return []
    scored = sorted(
        inactive, key=lambda i: _useful_score(aisles[i], unmet_demand), reverse=True
    )
    out = []
    for idx in scored[:cap]:
        if _useful_score(aisles[idx], unmet_demand) <= 0:
            break
        cand = mask.copy()
        cand[idx] = 1
        out.append(cand)
    return out


def _set_jaccard(s1: frozenset, s2: frozenset) -> float:
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1) + len(s2) - inter
    return inter / union if union else 0.0


def swap_neighbors(
    mask: np.ndarray,
    aisles: list[dict[int, int]],
    served_demand: dict[int, int],
    cap: int,
    aisle_key_sets: list[frozenset] | None = None,
) -> Iterable[np.ndarray]:
    """Generate masks with one active aisle replaced by an inactive one.

    Pairs are formed by: (active aisle with lowest utility) × (inactive aisles
    most similar to it). Capped to `cap` total candidates.

    `aisle_key_sets`, when provided, supplies precomputed `frozenset` views of
    each aisle's item keys, avoiding rebuilding them inside `similarity`.
    """
    active = np.flatnonzero(mask).tolist()
    inactive = np.flatnonzero(mask == 0).tolist()
    if not active or not inactive:
        return []

    active_sorted = sorted(active, key=lambda i: _useful_score(aisles[i], served_demand))

    # Pre-rank inactive by raw size as a coarse proxy; refine per-active by similarity.
    inactive_sorted = sorted(inactive, key=lambda i: _aisle_units(aisles[i]), reverse=True)
    inactive_pool = inactive_sorted[: max(cap, 20)]

    if aisle_key_sets is not None:
        sim = lambda a, b: _set_jaccard(aisle_key_sets[a], aisle_key_sets[b])
    else:
        sim = lambda a, b: similarity(aisles[a], aisles[b])

    out = []
    for a in active_sorted:
        if len(out) >= cap:
            break
        ranked_b = sorted(inactive_pool, key=lambda b: sim(a, b), reverse=True)
        for b in ranked_b:
            if len(out) >= cap:
                break
            cand = mask.copy()
            cand[a] = 0
            cand[b] = 1
            out.append(cand)
    return out


def _compute_demand_state(
    selected_orders: list[int],
    all_orders: list[dict[int, int]],
    n_orders: int,
    total_demand: dict[int, int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    served: dict[int, int] = {}
    for o_idx in selected_orders:
        for item, qty in all_orders[o_idx].items():
            served[item] = served.get(item, 0) + qty

    if total_demand is not None:
        # Fast path: unmet = total_demand - served. Iterates only items that
        # appear at all (typically << n_orders × items_per_order).
        unmet: dict[int, int] = {}
        for item, total in total_demand.items():
            remaining = total - served.get(item, 0)
            if remaining > 0:
                unmet[item] = remaining
        return served, unmet

    unmet = {}
    selected_set = set(selected_orders)
    for o_idx in range(n_orders):
        if o_idx in selected_set:
            continue
        for item, qty in all_orders[o_idx].items():
            unmet[item] = unmet.get(item, 0) + qty
    return served, unmet


def local_search_pass(
    mask: np.ndarray,
    fitness_fn: Callable[[np.ndarray], float],
    aisles: list[dict[int, int]],
    orders: list[dict[int, int]],
    operators: list[str],
    strategy: str,
    max_iterations: int,
    neighbor_cap: int,
    deadline: float | None,
    selected_orders_fn: Callable[[np.ndarray], list[int]] | None = None,
    total_demand: dict[int, int] | None = None,
    aisle_key_sets: list[frozenset] | None = None,
) -> tuple[np.ndarray, float, int]:
    """Run a hill-climb over the aisle mask until no improvement, time, or moves cap.

    Returns `(best_mask, best_obj, moves_evaluated)`. `fitness_fn` must already
    incorporate LB penalty / pruning so we can compare raw objectives.

    `total_demand` and `aisle_key_sets`, when provided, accelerate inner loops
    (unmet-demand subtraction and swap-neighbor similarity, respectively).
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"local_search: invalid strategy={strategy!r}")
    for op in operators:
        if op not in VALID_OPERATORS:
            raise ValueError(f"local_search: invalid operator={op!r}")

    def time_ok() -> bool:
        return deadline is None or time.time() < deadline

    best_mask = mask.copy()
    best_obj = fitness_fn(best_mask)
    moves = 0

    while moves < max_iterations and time_ok():
        served, unmet = _compute_demand_state(
            selected_orders_fn(best_mask) if selected_orders_fn else [],
            orders,
            len(orders),
            total_demand=total_demand,
        )

        improved = False
        round_best_mask = best_mask
        round_best_obj = best_obj

        for op_name in operators:
            if op_name == "remove":
                neighbors = remove_neighbors(best_mask, aisles, served, neighbor_cap)
            elif op_name == "add":
                neighbors = add_neighbors(best_mask, aisles, unmet, neighbor_cap)
            else:
                neighbors = swap_neighbors(
                    best_mask, aisles, served, neighbor_cap, aisle_key_sets
                )

            for cand in neighbors:
                if not time_ok() or moves >= max_iterations:
                    break
                obj = fitness_fn(cand)
                moves += 1
                if obj > round_best_obj + 1e-9:
                    round_best_obj = obj
                    round_best_mask = cand
                    improved = True
                    if strategy == "first_improvement":
                        break
            if improved and strategy == "first_improvement":
                break
            if not time_ok() or moves >= max_iterations:
                break

        if not improved:
            break
        best_mask = round_best_mask
        best_obj = round_best_obj

    return best_mask, best_obj, moves
