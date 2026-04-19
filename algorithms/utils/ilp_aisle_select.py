"""
ILP-based exact aisle selector for the wave order-picking problem.

Provides two formulations:
  1. solve_min_aisle_cover() — Set Multicover ILP (binary x_a only).
     Fastest for this problem; use as the primary drop-in for greedy selection.
  2. solve_cscp() — General Capacitated Set Cover Problem with explicit
     assignment variables y_ia and per-set capacity constraints.

Both use OR-Tools CP-SAT, which is already a project dependency.
"""

from __future__ import annotations

import time
from typing import NamedTuple

from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class ILPResult(NamedTuple):
    """Result returned by ILP aisle selectors."""

    selected_aisles: list[int]
    """Indices of aisles chosen by the solver."""

    status: str
    """Solver status: 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN', etc."""

    objective: int
    """Number of aisles selected (= objective value)."""

    solve_time: float
    """Wall-clock seconds spent inside the solver."""


# ---------------------------------------------------------------------------
# Formulation 1 — Minimum Set Multicover (recommended for this project)
# ---------------------------------------------------------------------------


def solve_min_aisle_cover(
    demand: dict[int, int],
    aisles: list[dict[int, int]],
    time_limit_seconds: float = 30.0,
) -> ILPResult:
    """Exact ILP: select the fewest aisles that collectively cover all demand.

    Model
    -----
    Variables   x_a ∈ {0, 1}  — 1 if aisle a is selected

    Minimize    Σ_a  x_a

    Subject to  Σ_a  supply[a][i] · x_a  ≥  demand[i]   ∀ item i with demand[i] > 0

    This is a Set Multicover ILP. Per-item supply bounds implicitly encode
    the capacity of each aisle, so no additional y variables are needed.

    Args:
        demand: {item_id: quantity_needed}  — aggregated from selected orders.
        aisles: list of {item_id: quantity_available} — one dict per aisle.
        time_limit_seconds: CP-SAT wall-clock limit.

    Returns:
        ILPResult with optimal (or best-found) aisle selection.
    """
    active_demand = {item: qty for item, qty in demand.items() if qty > 0}

    if not active_demand:
        return ILPResult([], "OPTIMAL", 0, 0.0)

    n_aisles = len(aisles)
    model = cp_model.CpModel()

    # Decision variables: x[a] = 1 iff aisle a is visited
    x = [model.new_bool_var(f"x_{a}") for a in range(n_aisles)]

    # Coverage constraints: total supply across selected aisles must meet demand
    for item, qty in active_demand.items():
        supply_terms = [aisles[a].get(item, 0) * x[a] for a in range(n_aisles)]
        model.add(cp_model.LinearExpr.sum(supply_terms) >= qty)

    # Objective: minimize number of visited aisles
    model.minimize(cp_model.LinearExpr.sum(x))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds

    t0 = time.perf_counter()
    status_code = solver.solve(model)
    elapsed = time.perf_counter() - t0

    status_name = solver.status_name(status_code)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [a for a in range(n_aisles) if solver.value(x[a]) == 1]
        return ILPResult(selected, status_name, len(selected), elapsed)

    # No feasible solution found within the time limit → greedy fallback
    from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select

    fallback = multi_greedy_aisle_select(dict(active_demand), aisles)
    return ILPResult(fallback, status_name, len(fallback), elapsed)


# ---------------------------------------------------------------------------
# Formulation 2 — General Capacitated Set Cover Problem (CSCP)
# ---------------------------------------------------------------------------


def solve_cscp(
    elements: list[int],
    subsets: list[list[int]],
    costs: list[float],
    capacities: list[int],
    time_limit_seconds: float = 30.0,
) -> ILPResult:
    """Exact ILP for the general Capacitated Set Cover Problem (CSCP).

    Model
    -----
    Variables   x_j ∈ {0, 1}    — 1 if subset j is selected
                y_ij ∈ {0, 1}   — 1 if element i is covered by subset j

    Minimize    Σ_j  cost_j · x_j

    Subject to
        Coverage:   Σ_j  y_ij  ≥  1            ∀ element i
        Capacity:   Σ_i  y_ij  ≤  capacity_j    ∀ subset j
        Linking:    y_ij  ≤  x_j                ∀ (i, j)  with i ∈ subset j

    Args:
        elements:    List of element identifiers (e.g. [0, 1, 2, ...]).
        subsets:     subsets[j] = list of elements that subset j can cover.
        costs:       costs[j] = cost of selecting subset j.
        capacities:  capacities[j] = max elements subset j can cover (k_j).
        time_limit_seconds: CP-SAT wall-clock limit.

    Returns:
        ILPResult where `selected_aisles` holds indices of selected subsets.

    Example
    -------
    >>> elements = [0, 1, 2, 3, 4]
    >>> subsets = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 4]]
    >>> costs = [3.0, 2.0, 2.0, 1.5]
    >>> capacities = [2, 2, 2, 2]
    >>> result = solve_cscp(elements, subsets, costs, capacities)
    >>> print(result.status, result.selected_aisles, result.objective)
    """
    n_elements = len(elements)
    n_subsets = len(subsets)

    # Map element id → position index for O(1) lookup
    elem_index = {e: i for i, e in enumerate(elements)}

    # For each element, which subsets can cover it?
    covering_subsets: list[list[int]] = [[] for _ in range(n_elements)]
    for j, subset in enumerate(subsets):
        for e in subset:
            if e in elem_index:
                covering_subsets[elem_index[e]].append(j)

    model = cp_model.CpModel()

    # x[j] = 1 if subset j is selected
    x = [model.new_bool_var(f"x_{j}") for j in range(n_subsets)]

    # y[i][j] = 1 if element i is covered by subset j
    # Only create y_ij for (i, j) pairs where j covers i (sparse)
    y: list[dict[int, cp_model.IntVar]] = [{} for _ in range(n_elements)]
    for i in range(n_elements):
        for j in covering_subsets[i]:
            y[i][j] = model.new_bool_var(f"y_{i}_{j}")

    # Coverage constraint: every element must be covered by at least one selected subset
    for i in range(n_elements):
        if not y[i]:
            raise ValueError(f"Element {elements[i]} cannot be covered by any subset.")
        model.add(cp_model.LinearExpr.sum(list(y[i].values())) >= 1)

    # Capacity constraint: each subset covers at most capacity_j elements
    for j in range(n_subsets):
        assigned_to_j = [y[i][j] for i in range(n_elements) if j in y[i]]
        if assigned_to_j:
            model.add(cp_model.LinearExpr.sum(assigned_to_j) <= capacities[j])

    # Linking constraint: y_ij = 1 implies x_j = 1
    for i in range(n_elements):
        for j, var in y[i].items():
            model.add(var <= x[j])

    # Objective: minimize total cost of selected subsets
    # CP-SAT requires integer coefficients; scale floats by 1000 for precision
    scale = 1000
    int_costs = [round(c * scale) for c in costs]
    model.minimize(
        cp_model.LinearExpr.weighted_sum(x, int_costs)
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds

    t0 = time.perf_counter()
    status_code = solver.solve(model)
    elapsed = time.perf_counter() - t0

    status_name = solver.status_name(status_code)

    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [j for j in range(n_subsets) if solver.value(x[j]) == 1]
        total_cost_scaled = solver.objective_value
        return ILPResult(selected, status_name, int(total_cost_scaled // scale), elapsed)

    return ILPResult([], status_name, 0, elapsed)


# ---------------------------------------------------------------------------
# Public drop-in replacement
# ---------------------------------------------------------------------------


def ilp_aisle_select(
    demand: dict[int, int],
    aisles: list[dict[int, int]],
    time_limit_seconds: float = 30.0,
) -> list[int]:
    """Select minimum aisles to cover demand using exact ILP.

    Drop-in replacement for ``multi_greedy_aisle_select()``.
    Falls back to the greedy solver if CP-SAT does not find a solution
    within ``time_limit_seconds``.

    Args:
        demand: {item_id: quantity_needed} — aggregated from selected orders.
        aisles: list of {item_id: quantity_available}.
        time_limit_seconds: solver time budget.

    Returns:
        List of selected aisle indices (order not guaranteed).
    """
    return solve_min_aisle_cover(demand, aisles, time_limit_seconds).selected_aisles


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import os
    import time as _time

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select

    print("=" * 60)
    print("Demo 1: solve_min_aisle_cover (Set Multicover ILP)")
    print("=" * 60)

    demand = {0: 5, 1: 3, 2: 2}
    aisles_demo = [
        {0: 3, 1: 2},           # aisle 0: partial
        {0: 2, 1: 1, 2: 2},     # aisle 1: partial
        {0: 5, 1: 3, 2: 2},     # aisle 2: covers everything alone
        {0: 1},                  # aisle 3: irrelevant
    ]

    t0 = _time.perf_counter()
    greedy_sol = multi_greedy_aisle_select(dict(demand), aisles_demo)
    greedy_time = _time.perf_counter() - t0
    print(f"Greedy : aisles={greedy_sol}  count={len(greedy_sol)}  time={greedy_time:.4f}s")

    result = solve_min_aisle_cover(demand, aisles_demo)
    print(f"ILP    : aisles={result.selected_aisles}  count={result.objective}"
          f"  status={result.status}  time={result.solve_time:.4f}s")

    print()
    print("=" * 60)
    print("Demo 2: solve_cscp (General CSCP with y_ij variables)")
    print("=" * 60)

    elements = [0, 1, 2, 3, 4]
    subsets = [
        [0, 1, 2],   # subset 0
        [1, 2, 3],   # subset 1
        [2, 3, 4],   # subset 2
        [0, 4],      # subset 3
        [0, 1, 2, 3, 4],  # subset 4: covers all, but expensive
    ]
    costs = [1.0, 1.0, 1.0, 1.0, 3.0]
    capacities = [2, 2, 2, 2, 5]

    cscp_result = solve_cscp(elements, subsets, costs, capacities)
    print(f"Selected subsets : {cscp_result.selected_aisles}")
    print(f"Total cost       : {sum(costs[j] for j in cscp_result.selected_aisles):.1f}")
    print(f"Status           : {cscp_result.status}")
    print(f"Solve time       : {cscp_result.solve_time:.4f}s")
