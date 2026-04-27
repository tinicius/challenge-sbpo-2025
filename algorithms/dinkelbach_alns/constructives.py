"""Constructive heuristics: produce a feasible warm-start WaveSolution.

Three strategies are run; the best feasible ratio wins.
  1. seed_by_density — pick a high-density seed order, expand greedily by Δratio.
  2. savings         — Clarke-Wright style: merge order pairs by aisle overlap.
  3. greedy_ratio    — pure greedy by marginal ratio.

Performance notes
-----------------
For large instances (N≈10k orders, A≈400 aisles), naïve O(N²) pair
enumeration with full min_cover per pair is intractable. We use:
  * cheap density proxy `units(o) / max(1, |items(o)|)` for ranking.
  * single-order covers are computed lazily and cached.
  * pair-savings approximation = |cover(o1)| + |cover(o2)| − |cover(o1)∪cover(o2)|
    (no full set-cover per pair).
  * exploration scoped to a top-K by density when N is large.
"""

from __future__ import annotations

import time
from typing import Iterable

from algorithms.utils.aisle_rank import (
    aggregate_demand,
    build_order_sequence,
    pack_orders,
    rank_aisles,
)
from problems.base import ProblemInput

from .preprocess import Preprocessed
from .state import WaveSolution


# ----------------------------------------------------------------------- #
#  Set-cover (full and incremental)                                       #
# ----------------------------------------------------------------------- #

def min_cover(
    order_set: Iterable[int],
    instance: ProblemInput,
    pre: Preprocessed,
) -> set[int]:
    demand: dict[int, int] = {}
    for o in order_set:
        for item, qty in instance.orders[o].items():
            demand[item] = demand.get(item, 0) + qty
    return _greedy_cover(demand, current_aisles=set(), instance=instance, pre=pre)


def min_cover_incremental(
    extra_demand: dict[int, int],
    current_aisles: set[int],
    instance: ProblemInput,
    pre: Preprocessed,
) -> set[int]:
    if not extra_demand:
        return set()
    return _greedy_cover(
        extra_demand, current_aisles=current_aisles, instance=instance, pre=pre
    )


def _greedy_cover(
    remaining: dict[int, int],
    current_aisles: set[int],
    instance: ProblemInput,
    pre: Preprocessed,
) -> set[int]:
    remaining = {item: qty for item, qty in remaining.items() if qty > 0}
    if not remaining:
        return set()
    active_set = set(pre.active_aisles) - current_aisles
    chosen: set[int] = set()

    candidates: set[int] = set()
    for item in remaining:
        for a in pre.item_to_aisles[item]:
            if a in active_set:
                candidates.add(a)

    while remaining:
        best_a = -1
        best_score = 0
        for a in candidates:
            a_dict = instance.aisles[a]
            score = 0
            for item, need in remaining.items():
                supply = a_dict.get(item, 0)
                if supply > 0:
                    score += supply if supply <= need else need
            if score > best_score:
                best_score = score
                best_a = a
        if best_a < 0 or best_score == 0:
            break
        chosen.add(best_a)
        candidates.discard(best_a)
        for item, qty in instance.aisles[best_a].items():
            if item in remaining:
                remaining[item] -= qty
                if remaining[item] <= 0:
                    del remaining[item]

    return chosen


# ----------------------------------------------------------------------- #
#  Density proxy + cached single-order covers                             #
# ----------------------------------------------------------------------- #

def _density_proxy(o: int, instance: ProblemInput, pre: Preprocessed) -> float:
    n_items = max(1, len(pre.order_to_items[o]))
    return float(pre.order_units[o]) / n_items


class _SingleOrderCoverCache:
    """Lazy cache: cover[o] computed on first request."""

    def __init__(self, instance: ProblemInput, pre: Preprocessed):
        self._inst = instance
        self._pre = pre
        self._cache: dict[int, set[int]] = {}

    def get(self, o: int) -> set[int]:
        c = self._cache.get(o)
        if c is None:
            c = min_cover([o], self._inst, self._pre)
            self._cache[o] = c
        return c


# ----------------------------------------------------------------------- #
#  Constructive 1: seed-by-density                                        #
# ----------------------------------------------------------------------- #

def seed_by_density(
    instance: ProblemInput,
    pre: Preprocessed,
    max_seeds: int = 3,
    cover_cache: _SingleOrderCoverCache | None = None,
) -> WaveSolution | None:
    n_orders = instance.nOrders
    if cover_cache is None:
        cover_cache = _SingleOrderCoverCache(instance, pre)

    densities = [
        (o, _density_proxy(o, instance, pre)) for o in range(n_orders)
    ]
    densities.sort(key=lambda t: -t[1])

    best: WaveSolution | None = None
    seeds_tried = 0
    for seed_o, _ in densities:
        if seeds_tried >= max_seeds:
            break
        if pre.order_units[seed_o] > instance.ub:
            continue
        seeds_tried += 1
        seed_cover = cover_cache.get(seed_o)
        sol = WaveSolution.from_sets({seed_o}, seed_cover, instance)
        if sol.total_units > instance.ub:
            continue
        sol = _expand_by_delta_ratio(sol, instance, pre)
        if sol is None or not sol.is_feasible(instance):
            continue
        if best is None or sol.ratio() > best.ratio():
            best = sol
    return best


def _expand_by_delta_ratio(
    sol: WaveSolution,
    instance: ProblemInput,
    pre: Preprocessed,
    candidate_pool_cap: int = 500,
    time_cap_s: float = 6.0,
) -> WaveSolution | None:
    """Greedy: at each step pick the order with the largest Δratio (or, if
    LB not yet met, the largest order that fits).

    For instances with many orders, the candidate pool is capped at
    `candidate_pool_cap` orders ranked by density proxy. When LB has not
    been reached and the dense pool is exhausted, the pool is widened to all
    orders ranked by units (descending) to give the LB-fallback a chance.
    """
    n_orders = instance.nOrders
    if n_orders <= candidate_pool_cap:
        dense_pool = list(range(n_orders))
    else:
        dense_pool = sorted(
            range(n_orders), key=lambda o: -_density_proxy(o, instance, pre)
        )[:candidate_pool_cap]
    candidate_pool = dense_pool
    widened = False

    deadline = time.time() + time_cap_s

    while True:
        if time.time() > deadline:
            break
        current_ratio = sol.ratio()
        best_o = -1
        best_extra: set[int] = set()
        best_new_ratio = current_ratio
        best_units_alt = -1
        best_o_alt = -1
        best_extra_alt: set[int] = set()

        for o in candidate_pool:
            if o in sol.orders:
                continue
            units_o = int(pre.order_units[o])
            if sol.total_units + units_o > instance.ub:
                continue

            extra_demand: dict[int, int] = {}
            for item, qty in instance.orders[o].items():
                gap = (sol.item_demand.get(item, 0) + qty) - sol.item_covered.get(item, 0)
                if gap > 0:
                    extra_demand[item] = gap
            extra = min_cover_incremental(extra_demand, sol.aisles, instance, pre)
            if extra_demand:
                # Verify that the cover actually closed every gap.
                supply_with_extra = dict(sol.item_covered)
                for a in extra:
                    for item, qty in instance.aisles[a].items():
                        if item in extra_demand:
                            supply_with_extra[item] = supply_with_extra.get(item, 0) + qty
                ok = all(
                    supply_with_extra.get(item, 0)
                    >= sol.item_demand.get(item, 0) + instance.orders[o].get(item, 0)
                    for item in extra_demand
                )
                if not ok:
                    continue

            new_aisles_count = len(sol.aisles) + len(extra)
            new_units = sol.total_units + units_o
            if new_aisles_count == 0:
                continue
            new_ratio = new_units / new_aisles_count

            if new_ratio > best_new_ratio + 1e-12:
                best_new_ratio = new_ratio
                best_o = o
                best_extra = extra

            if sol.total_units < instance.lb:
                if units_o > best_units_alt:
                    best_units_alt = units_o
                    best_o_alt = o
                    best_extra_alt = extra

        if best_o >= 0:
            _apply_add(sol, best_o, best_extra, instance)
            continue
        if sol.total_units < instance.lb:
            if best_o_alt >= 0:
                _apply_add(sol, best_o_alt, best_extra_alt, instance)
                continue
            # Widen the candidate pool once: scan all orders ranked by units.
            if not widened and n_orders > candidate_pool_cap:
                widened = True
                seen = set(dense_pool) | sol.orders
                extra = sorted(
                    (o for o in range(n_orders) if o not in seen),
                    key=lambda o: -int(pre.order_units[o]),
                )[:candidate_pool_cap]
                candidate_pool = extra
                continue
            return None
        break
    return sol


# ----------------------------------------------------------------------- #
#  Constructive 2: savings (Clarke-Wright adapted)                        #
# ----------------------------------------------------------------------- #

def savings_heuristic(
    instance: ProblemInput,
    pre: Preprocessed,
    top_orders: int = 300,
    max_pairs: int = 2000,
    cover_cache: _SingleOrderCoverCache | None = None,
) -> WaveSolution | None:
    """Pair-savings via precomputed single-order covers (set-union, not full
    set-cover per pair). Restricted to top-K orders by density when N is large."""
    n_orders = instance.nOrders
    if cover_cache is None:
        cover_cache = _SingleOrderCoverCache(instance, pre)

    densities = sorted(
        range(n_orders), key=lambda o: -_density_proxy(o, instance, pre)
    )
    candidates = densities[: min(top_orders, n_orders)]

    # Compute single-order covers only for the candidate set.
    covers = {o: cover_cache.get(o) for o in candidates}

    pair_savings: list[tuple[int, int, int, set[int]]] = []  # (savings, o1, o2, union_cover)
    for i, o1 in enumerate(candidates):
        c1 = covers[o1]
        for o2 in candidates[i + 1:]:
            c2 = covers[o2]
            union = c1 | c2
            save = len(c1) + len(c2) - len(union)
            if save > 0:
                pair_savings.append((save, o1, o2, union))

    pair_savings.sort(key=lambda t: -t[0])
    pair_savings = pair_savings[:max_pairs]

    selected: set[int] = set()
    total_units = 0
    cum_demand: dict[int, int] = {}

    def _try_add(o: int) -> bool:
        nonlocal total_units
        if o in selected:
            return False
        u = int(pre.order_units[o])
        if total_units + u > instance.ub:
            return False
        order_items = instance.orders[o]
        for item, qty in order_items.items():
            if cum_demand.get(item, 0) + qty > pre.total_supply.get(item, 0):
                return False
        selected.add(o)
        total_units += u
        for item, qty in order_items.items():
            cum_demand[item] = cum_demand.get(item, 0) + qty
        return True

    for _, o1, o2, _ in pair_savings:
        _try_add(o1)
        _try_add(o2)
        if total_units >= instance.lb:
            break

    if total_units < instance.lb:
        ranked = sorted(
            (o for o in range(n_orders) if o not in selected),
            key=lambda o: -int(pre.order_units[o]),
        )
        for o in ranked:
            _try_add(o)
            if total_units >= instance.lb:
                break

    if total_units < instance.lb or not selected:
        return None

    aisles = min_cover(selected, instance, pre)
    sol = WaveSolution.from_sets(selected, aisles, instance)
    if not sol.is_feasible(instance):
        return None
    return sol


# ----------------------------------------------------------------------- #
#  Constructive 3: greedy by marginal ratio                               #
# ----------------------------------------------------------------------- #

def greedy_ratio(instance: ProblemInput, pre: Preprocessed) -> WaveSolution | None:
    sol = WaveSolution()
    return _expand_by_delta_ratio(sol, instance, pre)


# ----------------------------------------------------------------------- #
#  Constructive 4: aisle-first sweep                                      #
# ----------------------------------------------------------------------- #

def aisle_first_sweep(
    instance: ProblemInput,
    pre: Preprocessed,
    score: str = "useful",
    order: str = "desc",
) -> WaveSolution | None:
    """Mirror of AisleFirstHeuristic: rank aisles by score, sweep k=1..K
    cumulatively, and pack orders into each prefix of size k. Tracks the
    k* that maximizes total_units / k. Restricted to aisles surviving the
    dominance prune so search space matches the rest of the pipeline.

    For the high-density A-instances this routinely lands within 92-100% of
    the best known objective in under 0.3s, providing a high-λ warm start
    that the rest of the ALNS layers cannot reach on their own."""
    n_orders = instance.nOrders
    n_aisles = instance.nAisles
    if n_aisles == 0 or n_orders == 0:
        return None

    order_sizes = [int(pre.order_units[o]) for o in range(n_orders)]
    total_demand = aggregate_demand(instance.orders)

    ranked = rank_aisles(instance.aisles, score, total_demand)
    active_set = set(pre.active_aisles)
    ranked = [a for a in ranked if a in active_set]
    if not ranked:
        return None

    seq = build_order_sequence(n_orders, order_sizes, order)

    best_k = 0
    best_orders: list[int] = []
    best_units = 0
    best_obj = 0.0

    inventory: dict[int, int] = {}
    for k, aisle_idx in enumerate(ranked, start=1):
        for item, qty in instance.aisles[aisle_idx].items():
            inventory[item] = inventory.get(item, 0) + qty

        if instance.ub / k <= best_obj:
            break

        selected, total_units = pack_orders(
            seq, instance.orders, order_sizes, inventory, instance.ub
        )
        if total_units < instance.lb:
            continue

        obj = total_units / k
        if obj > best_obj:
            best_obj = obj
            best_units = total_units
            best_orders = selected
            best_k = k

    if not best_orders or best_k == 0:
        return None

    sol = WaveSolution.from_sets(set(best_orders), set(ranked[:best_k]), instance)
    if not sol.is_feasible(instance):
        return None
    # Drop any aisle whose contribution is fully redundant (some prefix aisles
    # may not actually be needed once orders are packed).
    _prune_redundant_aisles_inplace(sol, instance)
    return sol


def _prune_redundant_aisles_inplace(
    sol: WaveSolution, instance: ProblemInput
) -> None:
    if len(sol.aisles) <= 1:
        return
    # Score aisles by how much demand they actually cover; drop the lowest
    # contributors whose removal still keeps coverage.
    scored = []
    for a in sol.aisles:
        contrib = 0
        for item, qty in instance.aisles[a].items():
            need = sol.item_demand.get(item, 0)
            if need <= 0:
                continue
            contrib += min(qty, need)
        scored.append((contrib, a))
    scored.sort()
    for _, a in scored:
        if len(sol.aisles) <= 1:
            break
        ok = True
        for item, qty in instance.aisles[a].items():
            if item not in sol.item_demand:
                continue
            if sol.item_covered.get(item, 0) - qty < sol.item_demand[item]:
                ok = False
                break
        if ok:
            sol.aisles.discard(a)
            for item, qty in instance.aisles[a].items():
                new_c = sol.item_covered.get(item, 0) - qty
                if new_c <= 0:
                    sol.item_covered.pop(item, None)
                else:
                    sol.item_covered[item] = new_c


# ----------------------------------------------------------------------- #
#  Helpers (also used later by ALNS operators)                            #
# ----------------------------------------------------------------------- #

def _apply_add(
    sol: WaveSolution, order: int, new_aisles: set[int], instance: ProblemInput
) -> None:
    sol.orders.add(order)
    sol.total_units += int(sum(instance.orders[order].values()))
    for item, qty in instance.orders[order].items():
        sol.item_demand[item] = sol.item_demand.get(item, 0) + qty
    for a in new_aisles:
        if a in sol.aisles:
            continue
        sol.aisles.add(a)
        for item, qty in instance.aisles[a].items():
            sol.item_covered[item] = sol.item_covered.get(item, 0) + qty


# ----------------------------------------------------------------------- #
#  Top-level: pick the best constructive                                  #
# ----------------------------------------------------------------------- #

def build_initial(
    instance: ProblemInput, pre: Preprocessed
) -> WaveSolution | None:
    feasible = build_initial_candidates(instance, pre)
    if feasible:
        return feasible[0]
    return _fallback_largest_orders(instance, pre)


def build_initial_candidates(
    instance: ProblemInput, pre: Preprocessed, max_candidates: int = 4
) -> list[WaveSolution]:
    """Return all distinct feasible warm-starts, ranked by ratio desc.

    Used by the multi-start ALNS layer: each candidate seeds an independent
    ALNS run, so different basins of attraction get explored in parallel
    instead of being collapsed into a single 'best' constructive."""
    cache = _SingleOrderCoverCache(instance, pre)
    candidates = [
        aisle_first_sweep(instance, pre, score="useful", order="desc"),
        seed_by_density(instance, pre, cover_cache=cache),
        savings_heuristic(instance, pre, cover_cache=cache),
        greedy_ratio(instance, pre),
    ]
    feasible = [s for s in candidates if s is not None and s.is_feasible(instance)]
    feasible.sort(key=lambda s: -s.ratio())
    deduped: list[WaveSolution] = []
    seen: set[frozenset[int]] = set()
    for s in feasible:
        key = frozenset(s.orders)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
        if len(deduped) >= max_candidates:
            break
    return deduped


def _fallback_largest_orders(
    instance: ProblemInput, pre: Preprocessed
) -> WaveSolution | None:
    """Same idea as the Fase-0 stub: pack largest orders that keep cumulative
    demand ≤ total_supply, then min_cover. Always tries to land within bounds."""
    selected: set[int] = set()
    total_units = 0
    cum_demand: dict[int, int] = {}
    ranked = sorted(range(instance.nOrders), key=lambda o: -int(pre.order_units[o]))
    for o in ranked:
        u = int(pre.order_units[o])
        if total_units + u > instance.ub:
            continue
        ok = True
        for item, qty in instance.orders[o].items():
            if cum_demand.get(item, 0) + qty > pre.total_supply.get(item, 0):
                ok = False
                break
        if not ok:
            continue
        selected.add(o)
        total_units += u
        for item, qty in instance.orders[o].items():
            cum_demand[item] = cum_demand.get(item, 0) + qty
        if total_units >= instance.lb:
            break
    if total_units < instance.lb:
        return None
    aisles = min_cover(selected, instance, pre)
    sol = WaveSolution.from_sets(selected, aisles, instance)
    if not sol.is_feasible(instance):
        return None
    return sol
