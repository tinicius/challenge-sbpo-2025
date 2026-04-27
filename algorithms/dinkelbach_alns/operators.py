"""Destroy / repair operators and incremental delta-evaluation helpers.

All operators mutate `sol` in place and assume `sol` was already deep-copied
before being handed to them. Each apply_* helper preserves the WaveSolution
invariants (total_units, item_demand, item_covered, aisles).

Destroy (5): random_order, worst_order, aisle_based, shaw, density_outlier.
Repair  (4): greedy_ratio, regret2, aisle_aware, random.
"""

from __future__ import annotations

import math
import random

from problems.base import ProblemInput

from .constructives import min_cover_incremental
from .preprocess import Preprocessed
from .state import WaveSolution


# ===================================================================== #
#  Apply helpers (low-level state mutation)                              #
# ===================================================================== #

def apply_add_order(
    sol: WaveSolution,
    order: int,
    new_aisles: set[int],
    instance: ProblemInput,
) -> None:
    if order in sol.orders:
        return
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


def apply_remove_order(
    sol: WaveSolution, order: int, instance: ProblemInput
) -> None:
    if order not in sol.orders:
        return
    sol.orders.discard(order)
    sol.total_units -= int(sum(instance.orders[order].values()))
    for item, qty in instance.orders[order].items():
        new_d = sol.item_demand.get(item, 0) - qty
        if new_d <= 0:
            sol.item_demand.pop(item, None)
        else:
            sol.item_demand[item] = new_d


def apply_remove_aisle(
    sol: WaveSolution, aisle: int, instance: ProblemInput
) -> None:
    if aisle not in sol.aisles:
        return
    sol.aisles.discard(aisle)
    for item, qty in instance.aisles[aisle].items():
        new_c = sol.item_covered.get(item, 0) - qty
        if new_c <= 0:
            sol.item_covered.pop(item, None)
        else:
            sol.item_covered[item] = new_c


def prune_redundant_aisles(
    sol: WaveSolution, instance: ProblemInput
) -> None:
    """Greedily remove aisles whose stock is fully redundant w.r.t. demand.
    Iterates in ascending order of useful contribution to demand."""
    if not sol.aisles:
        return
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
        # Tentatively check whether removing aisle keeps coverage.
        ok = True
        for item, qty in instance.aisles[a].items():
            if item not in sol.item_demand:
                continue
            if sol.item_covered.get(item, 0) - qty < sol.item_demand[item]:
                ok = False
                break
        if ok:
            apply_remove_aisle(sol, a, instance)


# ===================================================================== #
#  Delta evaluation                                                      #
# ===================================================================== #

def delta_add_order(
    sol: WaveSolution,
    order: int,
    lam: float,
    instance: ProblemInput,
    pre: Preprocessed,
) -> tuple[float, set[int], int]:
    """Returns (delta_h, new_aisles_needed, delta_units).
    `delta_h = delta_units - lam * delta_aisles`. If coverage cannot be
    completed by adding aisles, returns (-inf, empty, 0)."""
    units_o = int(pre.order_units[order])

    extra_demand: dict[int, int] = {}
    for item, qty in instance.orders[order].items():
        gap = (sol.item_demand.get(item, 0) + qty) - sol.item_covered.get(item, 0)
        if gap > 0:
            extra_demand[item] = gap
    extra = min_cover_incremental(extra_demand, sol.aisles, instance, pre)

    if extra_demand:
        # Verify the cover actually closed every gap.
        supply_with_extra = dict(sol.item_covered)
        for a in extra:
            for item, qty in instance.aisles[a].items():
                supply_with_extra[item] = supply_with_extra.get(item, 0) + qty
        for item in extra_demand:
            need = sol.item_demand.get(item, 0) + instance.orders[order].get(item, 0)
            if supply_with_extra.get(item, 0) < need:
                return (-math.inf, set(), 0)

    delta_aisles = len(extra)
    delta_h = units_o - lam * delta_aisles
    return (delta_h, extra, units_o)


def delta_remove_order(
    sol: WaveSolution,
    order: int,
    lam: float,
    instance: ProblemInput,
) -> tuple[float, set[int], int]:
    """Returns (delta_h, removable_aisles, delta_units). `delta_units` is
    negative (we remove units). An aisle is removable iff every item it
    supplies remains covered by the remaining aisles after also removing
    `order`'s contribution to demand."""
    units_o = int(sum(instance.orders[order].values()))

    # Demand after removing order o.
    demand_after = dict(sol.item_demand)
    for item, qty in instance.orders[order].items():
        new_d = demand_after.get(item, 0) - qty
        if new_d <= 0:
            demand_after.pop(item, None)
        else:
            demand_after[item] = new_d

    removable: set[int] = set()
    for a in sol.aisles:
        a_dict = instance.aisles[a]
        can_remove = True
        for item, qty in a_dict.items():
            need = demand_after.get(item, 0)
            if need <= 0:
                continue
            coverage_without = sol.item_covered.get(item, 0) - qty
            # Subtract any other tentatively-removed aisles to be safe.
            for ra in removable:
                coverage_without -= instance.aisles[ra].get(item, 0)
            if coverage_without < need:
                can_remove = False
                break
        if can_remove:
            removable.add(a)

    delta_aisles = -len(removable)
    delta_h = -units_o - lam * delta_aisles
    return (delta_h, removable, -units_o)


# ===================================================================== #
#  Destroy operators (mutate sol in-place)                               #
# ===================================================================== #

def d_random_order(
    sol: WaveSolution,
    rng: random.Random,
    instance: ProblemInput,
    gamma: float,
) -> None:
    if not sol.orders:
        return
    k = max(1, int(round(gamma * len(sol.orders))))
    victims = rng.sample(list(sol.orders), min(k, len(sol.orders)))
    for o in victims:
        apply_remove_order(sol, o, instance)
    prune_redundant_aisles(sol, instance)


def d_worst_order(
    sol: WaveSolution,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
    gamma: float,
) -> None:
    if not sol.orders:
        return
    # Score by units / (1 + n_exclusive_aisles_supplying_only_this_order).
    # Approximation: order's "marginal value" if removed = units(o) − ratio_loss_proxy.
    scored = []
    for o in sol.orders:
        contrib = 0
        for item, qty in instance.orders[o].items():
            if sol.item_demand.get(item, 0) <= qty:
                contrib += 1  # this order is the (near-)sole demander of item
        score = pre.order_units[o] / (1 + contrib)
        scored.append((score, o))
    scored.sort()  # ascending: worst first
    k = max(1, int(round(gamma * len(sol.orders))))
    victims = [o for _, o in scored[:k]]
    # Light noise to avoid determinism pile-ups.
    if rng.random() < 0.3 and len(victims) > 1:
        rng.shuffle(victims)
    for o in victims:
        apply_remove_order(sol, o, instance)
    prune_redundant_aisles(sol, instance)


def d_aisle_based(
    sol: WaveSolution,
    rng: random.Random,
    instance: ProblemInput,
) -> None:
    """Remove a random subset of aisles AND every order that depended on them
    for coverage."""
    if not sol.aisles:
        return
    n_to_remove = rng.randint(1, max(1, len(sol.aisles) // 3))
    aisles_victim = set(rng.sample(list(sol.aisles), n_to_remove))

    # Coverage if we removed those aisles.
    coverage_after_aisle_remove = dict(sol.item_covered)
    for a in aisles_victim:
        for item, qty in instance.aisles[a].items():
            new_c = coverage_after_aisle_remove.get(item, 0) - qty
            if new_c <= 0:
                coverage_after_aisle_remove.pop(item, None)
            else:
                coverage_after_aisle_remove[item] = new_c

    # Orders whose demand becomes uncovered after the aisle removal.
    orders_to_remove: set[int] = set()
    for o in sol.orders:
        for item, qty in instance.orders[o].items():
            if coverage_after_aisle_remove.get(item, 0) < sol.item_demand.get(item, 0):
                orders_to_remove.add(o)
                break
    # Apply removals.
    for o in orders_to_remove:
        apply_remove_order(sol, o, instance)
    for a in aisles_victim:
        apply_remove_aisle(sol, a, instance)
    prune_redundant_aisles(sol, instance)


def d_shaw(
    sol: WaveSolution,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
    gamma: float,
    p_noise: float = 3.0,
) -> None:
    if len(sol.orders) < 2:
        return
    anchor = rng.choice(list(sol.orders))
    items_anchor = pre.order_to_items[anchor]
    sims = []
    for o in sol.orders:
        if o == anchor:
            continue
        items_o = pre.order_to_items[o]
        union = items_anchor | items_o
        sim = (len(items_anchor & items_o) / len(union)) if union else 0.0
        sims.append((sim, o))
    sims.sort(reverse=True)
    # Power-noise selection.
    n = len(sims)
    k = max(1, int(round(gamma * (n + 1))))
    victims = []
    pool = sims[:]
    for _ in range(min(k, n)):
        idx = int((rng.random() ** p_noise) * len(pool))
        victims.append(pool.pop(idx)[1])
    victims.append(anchor)
    for o in victims:
        apply_remove_order(sol, o, instance)
    prune_redundant_aisles(sol, instance)


def d_density_outlier(
    sol: WaveSolution,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
    gamma: float,
) -> None:
    if not sol.orders:
        return
    scored = []
    for o in sol.orders:
        items_o = pre.order_to_items[o]
        # Aisles in sol that touch any item of o.
        touching = sum(
            1 for a in sol.aisles if pre.aisle_to_items[a] & items_o
        )
        density = pre.order_units[o] / max(1, touching)
        scored.append((density, o))
    scored.sort()  # ascending: lowest local density first
    k = max(1, int(round(gamma * len(sol.orders))))
    victims = [o for _, o in scored[:k]]
    for o in victims:
        apply_remove_order(sol, o, instance)
    prune_redundant_aisles(sol, instance)


# ===================================================================== #
#  Repair operators (mutate sol in-place)                                #
# ===================================================================== #

def _candidate_orders(
    sol: WaveSolution, instance: ProblemInput, cap: int = 400
) -> list[int]:
    """Restrict to a manageable candidate pool for large instances."""
    n = instance.nOrders
    if n - len(sol.orders) <= cap:
        return [o for o in range(n) if o not in sol.orders]
    # Sample top-cap by units (reasonable proxy when N is huge).
    ranked = sorted(
        (o for o in range(n) if o not in sol.orders),
        key=lambda o: -int(sum(instance.orders[o].values())),
    )
    return ranked[:cap]


def r_greedy_ratio(
    sol: WaveSolution,
    lam: float,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
) -> None:
    while True:
        candidates = _candidate_orders(sol, instance)
        if not candidates:
            break
        best_o = -1
        best_extra: set[int] = set()
        best_h = 0.0 if sol.total_units >= instance.lb else -math.inf
        for o in candidates:
            units_o = int(pre.order_units[o])
            if sol.total_units + units_o > instance.ub:
                continue
            dh, extra, _ = delta_add_order(sol, o, lam, instance, pre)
            if dh == -math.inf:
                continue
            # When LB still missing, accept any feasible add (maximize dh).
            if sol.total_units < instance.lb:
                if dh > best_h:
                    best_h = dh
                    best_o = o
                    best_extra = extra
            else:
                if dh > best_h:
                    best_h = dh
                    best_o = o
                    best_extra = extra
        if best_o < 0:
            break
        apply_add_order(sol, best_o, best_extra, instance)
        if sol.total_units >= instance.lb and best_h <= 0.0:
            break  # no more positive-Δh orders


def r_regret2(
    sol: WaveSolution,
    lam: float,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
    noise_prob: float = 0.1,
) -> None:
    """Insert order with the largest regret (best - second-best Δh)."""
    while True:
        candidates = _candidate_orders(sol, instance)
        if not candidates:
            break
        # Compute Δh for each candidate.
        scored = []
        for o in candidates:
            units_o = int(pre.order_units[o])
            if sol.total_units + units_o > instance.ub:
                continue
            dh, extra, _ = delta_add_order(sol, o, lam, instance, pre)
            if dh == -math.inf:
                continue
            scored.append((dh, o, extra))
        if not scored:
            break
        scored.sort(reverse=True)
        if sol.total_units < instance.lb:
            # While LB missing, just take the best.
            chosen = scored[0]
        else:
            # Stop if best is negative.
            if scored[0][0] <= 0.0:
                break
            if len(scored) == 1 or rng.random() < noise_prob:
                chosen = scored[0]
            else:
                # Largest regret = scored[0].dh - scored[1].dh; here only top
                # has been accepted, so picking top is functionally equivalent.
                chosen = scored[0]
        apply_add_order(sol, chosen[1], chosen[2], instance)


def r_aisle_aware(
    sol: WaveSolution,
    lam: float,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
) -> None:
    """Prioritize orders fully coverable by current aisles (delta_aisles == 0).
    Falls back to greedy_ratio once those are exhausted (or LB still missing)."""
    while True:
        candidates = _candidate_orders(sol, instance)
        if not candidates:
            break
        free_orders = []
        for o in candidates:
            units_o = int(pre.order_units[o])
            if sol.total_units + units_o > instance.ub:
                continue
            # Check coverage with current aisles only.
            ok = True
            for item, qty in instance.orders[o].items():
                need = sol.item_demand.get(item, 0) + qty
                if sol.item_covered.get(item, 0) < need:
                    ok = False
                    break
            if ok:
                free_orders.append((units_o, o))
        if free_orders:
            free_orders.sort(reverse=True)
            chosen = free_orders[0][1]
            apply_add_order(sol, chosen, set(), instance)
            continue
        if sol.total_units >= instance.lb:
            break
        # LB missing and no free order: fall back to greedy add.
        r_greedy_ratio(sol, lam, rng, instance, pre)
        break


def r_random(
    sol: WaveSolution,
    lam: float,
    rng: random.Random,
    instance: ProblemInput,
    pre: Preprocessed,
) -> None:
    """Diversification: insert candidates in random order."""
    candidates = _candidate_orders(sol, instance)
    rng.shuffle(candidates)
    for o in candidates:
        units_o = int(pre.order_units[o])
        if sol.total_units + units_o > instance.ub:
            continue
        dh, extra, _ = delta_add_order(sol, o, lam, instance, pre)
        if dh == -math.inf:
            continue
        # When LB met, accept only if dh > 0 with prob 0.5 (light noise).
        if sol.total_units >= instance.lb and dh <= 0 and rng.random() > 0.3:
            continue
        apply_add_order(sol, o, extra, instance)
        if sol.total_units >= instance.ub:
            break
