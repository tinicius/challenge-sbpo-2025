"""
Exact MIP solver via Dinkelbach's parametric algorithm + OR-Tools CP-SAT.

The fractional objective items/aisles is handled by converting it into
a sequence of linear 0-1 MIPs using Dinkelbach's method:

  maximize  q_d * Σ s_o*y_o  −  q_p * Σ x_a
  s.t.
    lb ≤ Σ s_o*y_o ≤ ub
    Σ_o demand_i(o)*y_o ≤ Σ_a supply_i(a)*x_a   ∀ item i
    y_o, x_a ∈ {0, 1}

where q = q_p / q_d is the current best ratio (updated each iteration).
Convergence is superlinear; typically 5–15 iterations suffice.
"""

import time

from ortools.sat.python import cp_model

from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.base import Algorithm
from problems.base import ProblemInput


class ExactMIP(Algorithm):

    @property
    def name(self) -> str:
        return "exact_mip"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        time_limit: float = float(self.params.get("time_limit", 590))
        max_iters: int = int(self.params.get("max_dinkelbach_iters", 20))
        num_workers: int = int(self.params.get("num_workers", 8))

        start_time = time.time()

        order_sizes = [sum(o.values()) for o in inst.orders]

        # Only items actually demanded by at least one order
        demanded_items: set[int] = set()
        for o in inst.orders:
            demanded_items.update(o.keys())

        # Prune aisles that supply no demanded item
        active_aisles = [
            a
            for a in range(inst.nAisles)
            if any(item in demanded_items for item in inst.aisles[a])
        ]

        # Warm-start: fast heuristic for initial ratio q = q_p / q_d
        best_sol = self._warmstart(inst)

        if best_sol["objective"] <= 0.0:
            q_p, q_d = 0, 1
        else:
            items_ws = sum(order_sizes[o] for o in best_sol["selected_orders"])
            aisles_ws = len(best_sol["visited_aisles"])
            q_p, q_d = items_ws, aisles_ws

        for iteration in range(max_iters):
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 1.0:
                break

            # Allocate time evenly over remaining iterations
            iter_budget = remaining / max(1, max_iters - iteration)

            sol = self._solve_cp_sat(
                inst,
                order_sizes,
                demanded_items,
                active_aisles,
                q_p,
                q_d,
                iter_budget,
                num_workers,
                best_sol,
            )

            if sol is None:
                break

            if sol["objective"] <= best_sol["objective"] + 1e-9:
                # Converged: ratio didn't improve
                break

            best_sol = sol
            q_p = sum(order_sizes[o] for o in sol["selected_orders"])
            q_d = len(sol["visited_aisles"])

        return best_sol

    def _warmstart(self, inst: ProblemInput) -> dict:
        heuristic = AisleFirstHeuristic(
            {"max_seeds": 5, "order_strategy": "size_desc"}
        )
        return heuristic.solve(inst)

    def _solve_cp_sat(
        self,
        inst: ProblemInput,
        order_sizes: list[int],
        demanded_items: set[int],
        active_aisles: list[int],
        q_p: int,
        q_d: int,
        time_budget: float,
        num_workers: int,
        hint: dict,
    ) -> dict | None:
        model = cp_model.CpModel()

        n_orders = inst.nOrders
        orders = inst.orders
        aisles = inst.aisles
        lb = inst.lb
        ub = inst.ub

        y = [model.NewBoolVar(f"y_{o}") for o in range(n_orders)]
        x = {a: model.NewBoolVar(f"x_{a}") for a in active_aisles}

        # Wave size bounds
        items_expr = sum(order_sizes[o] * y[o] for o in range(n_orders))
        model.Add(items_expr >= lb)
        model.Add(items_expr <= ub)

        # Stock coverage per item: demand ≤ supply from visited aisles
        for item in demanded_items:
            demand_terms = [
                orders[o][item] * y[o] for o in range(n_orders) if item in orders[o]
            ]
            if not demand_terms:
                continue

            supply_terms = [
                aisles[a][item] * x[a] for a in active_aisles if item in aisles[a]
            ]
            # If no aisle stocks this item, force all orders needing it to 0
            if not supply_terms:
                model.Add(sum(demand_terms) <= 0)
            else:
                model.Add(sum(demand_terms) <= sum(supply_terms))

        # Dinkelbach objective (integer scaled): maximize q_d*items − q_p*aisles
        aisle_count_expr = sum(x.values()) if x else 0
        model.Maximize(q_d * items_expr - q_p * aisle_count_expr)

        # Warm-start hints
        if hint.get("selected_orders") is not None:
            hint_orders = set(hint["selected_orders"])
            hint_aisles = set(hint["visited_aisles"])
            for o in range(n_orders):
                model.AddHint(y[o], 1 if o in hint_orders else 0)
            for a in active_aisles:
                model.AddHint(x[a], 1 if a in hint_aisles else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_budget
        solver.parameters.num_search_workers = num_workers

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None

        selected_orders = [o for o in range(n_orders) if solver.Value(y[o]) == 1]
        visited_aisles = [a for a in active_aisles if solver.Value(x[a]) == 1]

        if not visited_aisles:
            return None

        total_items = sum(order_sizes[o] for o in selected_orders)
        if total_items < lb:
            return None

        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": total_items / len(visited_aisles),
        }
