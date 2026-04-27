"""Local-search post-processing for aisle-first heuristics.

Wraps `local_search_pass` (remove/add/swap operators over an aisle mask) with a
fitness function that mirrors the AisleFirst objective: pack orders against the
mask's inventory, optionally apply prune over all aisles using the selected
orders' demand, return total_units / |visited|.

Used by `AisleFirstHeuristic` and `AisleFirstExactOrders` to refine the
constructed solution.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from algorithms.utils.aisle_rank import (
    aggregate_demand,
    aggregate_demand_from,
    pack_orders,
)
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.local_search_aisle import (
    VALID_OPERATORS,
    VALID_STRATEGIES,
    local_search_pass,
)
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_LS_DEFAULTS = {
    "operators": ["remove", "swap", "add"],
    "strategy": "first_improvement",
    "max_iterations": 200,
    "neighbor_cap": 50,
    "time_limit": None,
}

_LB_PENALTY_WEIGHT = 0.1


def validate_local_search(raw, owner: str) -> dict | None:
    """Normalize a `local_search` param into a config dict, or `None` if disabled.

    Accepts:
      - `None` / `False` → disabled
      - `True` → defaults
      - `dict` → defaults overridden by the dict's keys
    """
    if raw is None or raw is False:
        return None
    if raw is True:
        return dict(_LS_DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{owner}: 'local_search' must be bool or dict, got {type(raw).__name__}"
        )
    cfg = dict(_LS_DEFAULTS)
    cfg.update(raw)

    bad_ops = [o for o in cfg["operators"] if o not in VALID_OPERATORS]
    if bad_ops:
        raise ValueError(f"{owner}: invalid local_search operators {bad_ops}")
    if cfg["strategy"] not in VALID_STRATEGIES:
        raise ValueError(
            f"{owner}: invalid local_search.strategy={cfg['strategy']!r}; "
            f"expected one of {sorted(VALID_STRATEGIES)}"
        )
    if int(cfg["max_iterations"]) <= 0:
        raise ValueError(
            f"{owner}: 'local_search.max_iterations' must be > 0; "
            f"got {cfg['max_iterations']!r}"
        )
    if int(cfg["neighbor_cap"]) <= 0:
        raise ValueError(
            f"{owner}: 'local_search.neighbor_cap' must be > 0; "
            f"got {cfg['neighbor_cap']!r}"
        )
    if cfg["time_limit"] is not None and float(cfg["time_limit"]) <= 0:
        raise ValueError(
            f"{owner}: 'local_search.time_limit' must be > 0 or unset; "
            f"got {cfg['time_limit']!r}"
        )

    cfg["operators"] = list(cfg["operators"])
    cfg["max_iterations"] = int(cfg["max_iterations"])
    cfg["neighbor_cap"] = int(cfg["neighbor_cap"])
    cfg["time_limit"] = float(cfg["time_limit"]) if cfg["time_limit"] is not None else None
    return cfg


def _make_fitness(
    instance: ProblemInput,
    order_sequence: list[int],
    order_sizes: list[int],
    prune_mode: str | None,
):
    """Build a cached fitness function over a binary aisle mask.

    Cache entries are `(obj, selected_orders, visited_aisles)` keyed by the
    sorted tuple of active aisle indices. `selected_orders_fn` reads from this
    cache so LS can recover the order set for the final mask without repacking.
    """
    aisles = instance.aisles
    orders = instance.orders
    lb, ub = instance.lb, instance.ub
    cache: dict[tuple[int, ...], tuple[float, list[int], list[int]]] = {}

    if prune_mode == "multi":
        prune_fn = multi_greedy_aisle_select
    elif prune_mode == "simple":
        prune_fn = greedy_aisle_select
    else:
        prune_fn = None

    def fitness(mask: np.ndarray) -> float:
        active = np.flatnonzero(mask).tolist()
        if not active:
            return 0.0
        key = tuple(active)
        cached = cache.get(key)
        if cached is not None:
            return cached[0]

        inventory: dict[int, int] = {}
        for a in active:
            for item, qty in aisles[a].items():
                inventory[item] = inventory.get(item, 0) + qty

        selected, total_units = pack_orders(
            order_sequence, orders, order_sizes, inventory, ub
        )

        if total_units < lb:
            obj = (total_units / max(len(active), 1)) * _LB_PENALTY_WEIGHT
            cache[key] = (obj, [], [])
            return obj

        if prune_fn is not None:
            demand = aggregate_demand_from(orders, selected)
            visited = prune_fn(demand, aisles)
            if not visited:
                obj = (total_units / max(len(active), 1)) * _LB_PENALTY_WEIGHT
                cache[key] = (obj, selected, [])
                return obj
            obj = total_units / len(visited)
        else:
            visited = list(active)
            obj = total_units / len(active)

        cache[key] = (obj, selected, visited)
        return obj

    fitness.cache = cache  # type: ignore[attr-defined]
    return fitness


def apply_local_search(
    result: dict,
    instance: ProblemInput,
    prune_mode: str | None,
    ls_config: dict,
    order_sequence: list[int],
    order_sizes: list[int],
) -> dict:
    """Hill-climb the aisle mask of `result` using the configured operators.

    Returns the original `result` if LS cannot improve it.
    """
    visited = result.get("visited_aisles") or []
    if not visited:
        return result

    n_aisles = instance.nAisles
    mask = np.zeros(n_aisles, dtype=int)
    mask[np.asarray(visited, dtype=int)] = 1

    fitness = _make_fitness(instance, order_sequence, order_sizes, prune_mode)
    # Prime cache so LS starts from the incumbent's exact value.
    incumbent_obj = fitness(mask)

    total_demand = aggregate_demand(instance.orders)
    aisle_key_sets = [frozenset(a.keys()) for a in instance.aisles]

    deadline = (
        time.time() + ls_config["time_limit"]
        if ls_config.get("time_limit") is not None
        else None
    )

    def orders_for_mask(m: np.ndarray) -> list[int]:
        key = tuple(np.flatnonzero(m).tolist())
        cached = fitness.cache.get(key)
        if cached is None:
            fitness(m)
            cached = fitness.cache.get(key)
        return cached[1] if cached else []

    new_mask, new_obj, _ = local_search_pass(
        mask=mask,
        fitness_fn=fitness,
        aisles=instance.aisles,
        orders=instance.orders,
        operators=ls_config["operators"],
        strategy=ls_config["strategy"],
        max_iterations=ls_config["max_iterations"],
        neighbor_cap=ls_config["neighbor_cap"],
        deadline=deadline,
        selected_orders_fn=orders_for_mask,
        total_demand=total_demand,
        aisle_key_sets=aisle_key_sets,
    )

    if new_obj <= incumbent_obj + 1e-9:
        return result

    key = tuple(np.flatnonzero(new_mask).tolist())
    cached = fitness.cache.get(key)
    if cached is None:
        return result
    obj, selected, ls_visited = cached
    if not selected or not ls_visited or obj <= result["objective"] + 1e-9:
        return result

    return {
        "selected_orders": selected,
        "visited_aisles": ls_visited,
        "objective": obj,
    }
