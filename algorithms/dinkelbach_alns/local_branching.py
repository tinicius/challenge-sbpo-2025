"""CP-SAT Local Branching refinement.

Given an incumbent WaveSolution, search a Hamming-k neighborhood (in the order
selection vector) for a higher-ratio configuration. Solve once per k in
`k_values` (defaults to [5, 10, 20]); λ is refreshed from the incumbent ratio
between rounds (Dinkelbach progressive). The CP-SAT objective approximates
`units − λ·|A'|` via integer scaling (SCALE = 1000)."""

from __future__ import annotations

import time

from ortools.sat.python import cp_model

from problems.base import ProblemInput

from .preprocess import Preprocessed
from .state import WaveSolution

SCALE = 1000


def local_branching_refinement(
    sol: WaveSolution,
    instance: ProblemInput,
    pre: Preprocessed,
    lam: float,
    time_limit: float,
    k_values: list[int] | None = None,
    num_workers: int = 4,
    seed: int = 42,
) -> WaveSolution:
    """Polish `sol` using CP-SAT k-Hamming local branching.

    Returns the best feasible WaveSolution found (>= sol.ratio())."""
    if not sol.is_feasible(instance):
        return sol
    if k_values is None:
        k_values = [5, 10, 20]
    if time_limit <= 0:
        return sol

    best = sol.copy()
    current_lam = lam if lam > 0 else best.ratio()
    deadline = time.time() + time_limit

    for k in k_values:
        remaining = deadline - time.time()
        if remaining < 1.0:
            break
        per_round = max(2.0, remaining / max(1, len([kk for kk in k_values
                                                    if kk >= k])))
        new_sol, status = _solve_lb(
            best, instance, pre, current_lam, k, per_round, num_workers, seed
        )
        if new_sol is not None and new_sol.ratio() > best.ratio() + 1e-9:
            best = new_sol
            current_lam = best.ratio()
        if status == cp_model.INFEASIBLE:
            # Hamming-k box has no improving solution; widening k may help.
            continue

    return best


def _solve_lb(
    incumbent: WaveSolution,
    instance: ProblemInput,
    pre: Preprocessed,
    lam: float,
    k: int,
    time_limit: float,
    num_workers: int,
    seed: int,
) -> tuple[WaveSolution | None, int]:
    """Build and solve the local-branching CP-SAT model. Returns
    (best WaveSolution if improved else None, solver status)."""
    n_orders = instance.nOrders
    active_aisles = pre.active_aisles
    if not active_aisles:
        return None, cp_model.MODEL_INVALID

    model = cp_model.CpModel()

    x = [model.NewBoolVar(f"x_{o}") for o in range(n_orders)]
    y = {a: model.NewBoolVar(f"y_{a}") for a in active_aisles}

    # LB / UB on units.
    units_terms = [int(pre.order_units[o]) * x[o] for o in range(n_orders)]
    model.Add(sum(units_terms) >= int(instance.lb))
    model.Add(sum(units_terms) <= int(instance.ub))

    # Item coverage: Σ u_oi · x_o ≤ Σ u_ai · y_a, ∀ demanded item.
    demanded = set()
    for o in range(n_orders):
        demanded.update(instance.orders[o].keys())
    for item in demanded:
        order_terms = [
            instance.orders[o][item] * x[o]
            for o in pre.item_to_orders[item]
        ]
        if not order_terms:
            continue
        aisle_terms = [
            instance.aisles[a][item] * y[a]
            for a in pre.item_to_aisles[item]
            if a in y
        ]
        if not aisle_terms:
            # No active aisle supplies this item — force demand to 0.
            for term in order_terms:
                model.Add(term == 0)
            continue
        model.Add(sum(order_terms) <= sum(aisle_terms))

    # Hamming-k local branching constraint on x:
    # Σ_{o ∈ inc} (1 − x_o) + Σ_{o ∉ inc} x_o ≤ k.
    inc_orders = incumbent.orders
    flips = []
    for o in range(n_orders):
        if o in inc_orders:
            flips.append(1 - x[o])
        else:
            flips.append(x[o])
    model.Add(sum(flips) <= int(k))

    # Objective: SCALE · units − round(SCALE·λ) · |A'|.
    lam_int = int(round(SCALE * max(lam, 0.0)))
    units_expr = sum(int(pre.order_units[o]) * x[o] for o in range(n_orders))
    aisles_expr = sum(y[a] for a in active_aisles)
    model.Maximize(SCALE * units_expr - lam_int * aisles_expr)

    # Warm-start hint with incumbent.
    for o in range(n_orders):
        model.AddHint(x[o], 1 if o in inc_orders else 0)
    for a in active_aisles:
        model.AddHint(y[a], 1 if a in incumbent.aisles else 0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = max(1, int(num_workers))
    solver.parameters.random_seed = int(seed)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status

    selected_orders = {
        o for o in range(n_orders) if solver.Value(x[o]) == 1
    }
    selected_aisles = {a for a in active_aisles if solver.Value(y[a]) == 1}
    if not selected_aisles:
        return None, status

    new_sol = WaveSolution.from_sets(selected_orders, selected_aisles, instance)
    if not new_sol.is_feasible(instance):
        return None, status
    return new_sol, status
