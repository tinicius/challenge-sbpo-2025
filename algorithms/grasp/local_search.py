"""Order-level local-search moves shared between GRASP variants.

Each move takes a *solution dict* containing the private `_demand` and
`_total_units` keys (set during construction) plus the callable
`select_aisles(demand, aisles) -> list[int]` that chooses visited aisles for a
given demand. Moves use first-improvement: return a new solution dict if an
improving move is found, else None.
"""

from typing import Callable


def demand_within_stock(
    demand: dict[int, int], stock_total: dict[int, int]
) -> bool:
    for item, qty in demand.items():
        if stock_total.get(item, 0) < qty:
            return False
    return True


def try_order_swap(
    solution: dict,
    orders: list[dict[int, int]],
    aisles: list[dict[int, int]],
    order_sizes: list[int],
    stock_total: dict[int, int],
    lb: int,
    ub: int,
    select_aisles: Callable[[dict[int, int], list[dict[int, int]]], list[int]],
) -> dict | None:
    selected = solution["selected_orders"]
    selected_set = set(selected)
    demand = solution["_demand"]
    total_units = solution["_total_units"]
    base_obj = solution["objective"]
    n_orders = len(orders)

    for s_idx in selected:
        s_order = orders[s_idx]
        s_size = order_sizes[s_idx]
        demand_minus = dict(demand)
        for item, qty in s_order.items():
            demand_minus[item] = demand_minus.get(item, 0) - qty
            if demand_minus[item] == 0:
                del demand_minus[item]

        for u_idx in range(n_orders):
            if u_idx in selected_set:
                continue
            u_order = orders[u_idx]
            u_size = order_sizes[u_idx]
            if u_size == 0:
                continue
            new_total = total_units - s_size + u_size
            if new_total > ub or new_total < lb:
                continue

            new_demand = dict(demand_minus)
            for item, qty in u_order.items():
                new_demand[item] = new_demand.get(item, 0) + qty

            if not demand_within_stock(new_demand, stock_total):
                continue

            new_visited = select_aisles(new_demand, aisles)
            if not new_visited:
                continue
            new_obj = new_total / len(new_visited)
            if new_obj > base_obj:
                new_selected = [i for i in selected if i != s_idx]
                new_selected.append(u_idx)
                return {
                    "selected_orders": new_selected,
                    "visited_aisles": new_visited,
                    "objective": new_obj,
                    "_demand": new_demand,
                    "_total_units": new_total,
                }

    return None


def try_order_drop(
    solution: dict,
    orders: list[dict[int, int]],
    aisles: list[dict[int, int]],
    order_sizes: list[int],
    lb: int,
    select_aisles: Callable[[dict[int, int], list[dict[int, int]]], list[int]],
) -> dict | None:
    selected = solution["selected_orders"]
    demand = solution["_demand"]
    total_units = solution["_total_units"]
    base_obj = solution["objective"]

    if len(selected) <= 1:
        return None

    for s_idx in selected:
        s_order = orders[s_idx]
        s_size = order_sizes[s_idx]
        new_total = total_units - s_size
        if new_total < lb:
            continue

        new_demand = dict(demand)
        for item, qty in s_order.items():
            new_demand[item] = new_demand.get(item, 0) - qty
            if new_demand[item] == 0:
                del new_demand[item]

        new_visited = select_aisles(new_demand, aisles)
        if not new_visited:
            continue
        new_obj = new_total / len(new_visited)
        if new_obj > base_obj:
            new_selected = [i for i in selected if i != s_idx]
            return {
                "selected_orders": new_selected,
                "visited_aisles": new_visited,
                "objective": new_obj,
                "_demand": new_demand,
                "_total_units": new_total,
            }

    return None


def try_order_add(
    solution: dict,
    orders: list[dict[int, int]],
    aisles: list[dict[int, int]],
    order_sizes: list[int],
    stock_total: dict[int, int],
    ub: int,
    select_aisles: Callable[[dict[int, int], list[dict[int, int]]], list[int]],
) -> dict | None:
    selected = solution["selected_orders"]
    selected_set = set(selected)
    demand = solution["_demand"]
    total_units = solution["_total_units"]
    base_obj = solution["objective"]
    n_orders = len(orders)

    for u_idx in range(n_orders):
        if u_idx in selected_set:
            continue
        u_size = order_sizes[u_idx]
        if u_size == 0:
            continue
        new_total = total_units + u_size
        if new_total > ub:
            continue

        u_order = orders[u_idx]
        new_demand = dict(demand)
        for item, qty in u_order.items():
            new_demand[item] = new_demand.get(item, 0) + qty

        if not demand_within_stock(new_demand, stock_total):
            continue

        new_visited = select_aisles(new_demand, aisles)
        if not new_visited:
            continue
        new_obj = new_total / len(new_visited)
        if new_obj > base_obj:
            new_selected = list(selected)
            new_selected.append(u_idx)
            return {
                "selected_orders": new_selected,
                "visited_aisles": new_visited,
                "objective": new_obj,
                "_demand": new_demand,
                "_total_units": new_total,
            }

    return None


def improve_orders(
    solution: dict,
    orders: list[dict[int, int]],
    aisles: list[dict[int, int]],
    order_sizes: list[int],
    stock_total: dict[int, int],
    lb: int,
    ub: int,
    select_aisles: Callable[[dict[int, int], list[dict[int, int]]], list[int]],
    mode: str,
) -> dict:
    """Run order-level local search until no improving move is found.

    `mode` in {"swap", "full"}: "swap" uses only try_order_swap; "full" also
    runs try_order_drop and try_order_add after swap each pass. Same semantics
    as GraspHeuristic._local_search_improve.
    """
    current = solution
    improved = True
    while improved:
        improved = False

        swap_move = try_order_swap(
            current, orders, aisles, order_sizes, stock_total, lb, ub, select_aisles
        )
        if swap_move is not None:
            current = swap_move
            improved = True
            continue

        if mode == "full":
            drop_move = try_order_drop(
                current, orders, aisles, order_sizes, lb, select_aisles
            )
            if drop_move is not None:
                current = drop_move
                improved = True
                continue

            add_move = try_order_add(
                current, orders, aisles, order_sizes, stock_total, ub, select_aisles
            )
            if add_move is not None:
                current = add_move
                improved = True
                continue

    return current
