"""
ILP-based exact order selector for the wave order-picking problem.

Given a fixed inventory pool (from pre-selected aisles), selects the subset
of orders that maximises total items collected while satisfying the wave-size
bounds lb ≤ total_items ≤ ub and per-item inventory constraints.

Model
-----
Variables   y_o ∈ {0, 1}  — 1 if order o is selected

Maximise    Σ_o  size(o) · y_o

Subject to  lb  ≤  Σ_o  size(o) · y_o  ≤  ub
            Σ_o  demand[o][i] · y_o  ≤  inventory_pool[i]   ∀ item i
"""

from __future__ import annotations

import time

from ortools.sat.python import cp_model


def solve_max_order_select(
    inventory_pool: dict[int, int],
    orders: list[dict[int, int]],
    lb: int,
    ub: int,
    time_limit_seconds: float = 10.0,
) -> list[int]:
    """Select orders to maximise total items within the given inventory pool.

    Args:
        inventory_pool: {item_id: quantity_available} — pooled stock from
            pre-selected aisles.
        orders: list of {item_id: quantity_needed} — one dict per order.
        lb: minimum total items required (wave lower bound).
        ub: maximum total items allowed (wave upper bound).
        time_limit_seconds: CP-SAT wall-clock limit.

    Returns:
        List of selected order indices.  Empty if no feasible solution is
        found within the time limit.
    """
    n_orders = len(orders)

    # Pre-filter: drop orders that are infeasible against the current pool.
    # An order is infeasible if it demands more of any item than the pool holds.
    feasible = [
        o
        for o in range(n_orders)
        if all(inventory_pool.get(item, 0) >= qty for item, qty in orders[o].items())
    ]

    if not feasible:
        return []

    order_sizes = {o: sum(orders[o].values()) for o in feasible}

    model = cp_model.CpModel()
    y = {o: model.new_bool_var(f"y_{o}") for o in feasible}

    y_vars = [y[o] for o in feasible]
    size_coeffs = [order_sizes[o] for o in feasible]

    # Wave size bounds
    items_expr = cp_model.LinearExpr.weighted_sum(y_vars, size_coeffs)
    model.add(items_expr >= lb)
    model.add(items_expr <= ub)

    # Per-item inventory constraints
    for item, available in inventory_pool.items():
        demand_terms = [
            (orders[o][item], y[o]) for o in feasible if item in orders[o]
        ]
        if not demand_terms:
            continue
        coeffs, vars_ = zip(*demand_terms)
        model.add(
            cp_model.LinearExpr.weighted_sum(list(vars_), list(coeffs)) <= available
        )

    model.maximize(items_expr)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds

    status_code = solver.solve(model)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [o for o in feasible if solver.value(y[o]) == 1]

    return []
