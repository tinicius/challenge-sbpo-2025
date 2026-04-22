"""Shared aisle-centric greedy primitives.

Extracted from `aisle_first_heuristic.py` so that both `aisle_first` and
`aisle_grasp` can share the exact same ranking/packing core.
"""

import random

VALID_AISLE_SCORE = {"useful", "units", "variety", "mixed"}
VALID_ORDER_MODE = {None, "asc", "desc"}


def _score_fn(score: str, demand_gap: dict[int, int] | None):
    if score == "useful":
        gap = demand_gap or {}

        def fn(a: dict[int, int]) -> int:
            return sum(min(qty, gap.get(item, 0)) for item, qty in a.items())

        return fn
    if score == "units":
        return lambda a: sum(a.values())
    if score == "variety":
        return lambda a: len(a)
    if score == "mixed":
        return lambda a: sum(a.values()) * len(a)
    raise ValueError(
        f"aisle_rank: invalid score={score!r}; "
        f"expected one of {sorted(VALID_AISLE_SCORE)}"
    )


def rank_aisles(
    aisles: list[dict[int, int]],
    score: str,
    demand_gap: dict[int, int] | None,
) -> list[int]:
    """Return aisle indices sorted descending by `score` against `demand_gap`.

    `demand_gap` is only consulted by the "useful" score; other scores ignore it.
    """
    fn = _score_fn(score, demand_gap)
    return sorted(
        range(len(aisles)),
        key=lambda idx: fn(aisles[idx]),
        reverse=True,
    )


def score_aisles(
    aisles: list[dict[int, int]],
    indices,
    score: str,
    demand_gap: dict[int, int] | None,
) -> dict[int, float]:
    """Return {aisle_idx: score} for each idx in `indices` against `demand_gap`."""
    fn = _score_fn(score, demand_gap)
    return {idx: float(fn(aisles[idx])) for idx in indices}


def aggregate_demand(orders: list[dict[int, int]]) -> dict[int, int]:
    demand: dict[int, int] = {}
    for o in orders:
        for item, qty in o.items():
            demand[item] = demand.get(item, 0) + qty
    return demand


def aggregate_demand_from(
    orders: list[dict[int, int]], selected
) -> dict[int, int]:
    demand: dict[int, int] = {}
    for idx in selected:
        for item, qty in orders[idx].items():
            demand[item] = demand.get(item, 0) + qty
    return demand


def build_order_sequence(
    n_orders: int,
    order_sizes: list[int],
    order_mode,
    rng: random.Random | None = None,
    seed=None,
) -> list[int]:
    """Build the order packing sequence.

    - `order_mode="asc"` / `"desc"`: deterministic sort by order size.
    - `order_mode=None`: shuffled. Uses the provided `rng` if given,
      otherwise a `random.Random(seed)` when `seed is not None`,
      otherwise the global `random` module.
    """
    if order_mode not in VALID_ORDER_MODE:
        raise ValueError(
            f"aisle_rank: invalid order_mode={order_mode!r}; "
            f"expected one of {sorted(v for v in VALID_ORDER_MODE if v)} or None"
        )
    if order_mode is None:
        indices = list(range(n_orders))
        if rng is None:
            rng = random.Random(seed) if seed is not None else random
        rng.shuffle(indices)
        return indices
    return sorted(
        range(n_orders),
        key=order_sizes.__getitem__,
        reverse=(order_mode == "desc"),
    )


def pack_orders(
    sequence,
    orders: list[dict[int, int]],
    order_sizes: list[int],
    inventory: dict[int, int],
    ub: int,
) -> tuple[list[int], int]:
    """Greedy bin-pack orders from `sequence` into `inventory` under `ub`."""
    remaining = dict(inventory)
    selected: list[int] = []
    total = 0
    for idx in sequence:
        size = order_sizes[idx]
        if total + size > ub:
            continue
        order = orders[idx]
        if any(remaining.get(item, 0) < qty for item, qty in order.items()):
            continue
        selected.append(idx)
        total += size
        for item, qty in order.items():
            remaining[item] -= qty
    return selected, total
