"""
Approach C - Knapsack DP with Linear Aisle Penalty + Fractional Upper Bound.

Order value is modeled as:

    value(o) = total_items(o) - lambda_penalty * A(o)

where A(o) is the number of aisles needed to fulfill order o alone. This
encourages high-volume orders while penalizing aisle dispersion. A fractional
knapsack relaxation is also computed as an upper bound for this surrogate
order-selection objective.
"""

from algorithms.base import Algorithm
from algorithms.utils.knapsack_dp import knapsack_dp, repair_infeasible
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput
from problems.validation import is_solution_feasible


class KnapsackDPRelaxation(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)
        self.lambda_penalty = float(params.get("lambda_penalty", 1.0))

    @property
    def name(self) -> str:
        return "knapsack_dp_relaxation"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders, aisles = inst.orders, inst.aisles
        n_orders = inst.nOrders
        lb, ub = inst.lb, inst.ub

        order_sizes = [sum(order.values()) for order in orders]

        # Pre-compute A(o): aisles needed for each order individually.
        aisle_counts: list[int] = []
        for order in orders:
            if not order:
                aisle_counts.append(0)
                continue
            needed = multi_greedy_aisle_select(dict(order), aisles)
            aisle_counts.append(len(needed))

        weights: list[int] = []
        values: list[float] = []
        order_indices: list[int] = []

        for i in range(n_orders):
            if order_sizes[i] == 0 or aisle_counts[i] == 0:
                continue
            weights.append(order_sizes[i])
            values.append(order_sizes[i] - self.lambda_penalty * aisle_counts[i])
            order_indices.append(i)

        if not order_indices:
            return {
                "selected_orders": [],
                "visited_aisles": [],
                "objective": 0.0,
                "upper_bound": float("-inf"),
            }

        upper_bound = self._fractional_upper_bound(weights, values, ub, lb)
        positions = knapsack_dp(weights, values, ub, lb)

        if positions is None:
            return {
                "selected_orders": [],
                "visited_aisles": [],
                "objective": 0.0,
                "upper_bound": upper_bound,
            }

        selected_orders = [order_indices[p] for p in positions]
        selected_orders = repair_infeasible(
            orders, order_sizes, selected_orders, aisles, lb
        )
        if not selected_orders:
            return {
                "selected_orders": [],
                "visited_aisles": [],
                "objective": 0.0,
                "upper_bound": upper_bound,
            }

        selected_orders, visited_aisles = self._repair_until_feasible(
            inst,
            selected_orders,
            order_sizes,
            aisle_counts,
        )
        if not selected_orders:
            return {
                "selected_orders": [],
                "visited_aisles": [],
                "objective": 0.0,
                "upper_bound": upper_bound,
            }

        total_units = sum(order_sizes[o] for o in selected_orders)
        objective = total_units / len(visited_aisles)
        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": objective,
            "upper_bound": upper_bound,
        }

    def _repair_until_feasible(
        self,
        inst: ProblemInput,
        selected_orders: list[int],
        order_sizes: list[int],
        aisle_counts: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Remove orders iteratively until a fully feasible solution is found.
        Keeps removing the order contributing most to current stock deficits.
        """
        current = list(selected_orders)

        while current:
            total_units = sum(order_sizes[o] for o in current)
            if total_units < inst.lb:
                return [], []

            demand: dict[int, int] = {}
            for o in current:
                for item, qty in inst.orders[o].items():
                    demand[item] = demand.get(item, 0) + qty

            visited_aisles = multi_greedy_aisle_select(demand, inst.aisles)
            if visited_aisles and is_solution_feasible(inst, current, visited_aisles):
                return current, visited_aisles

            available: dict[int, int] = {}
            for a in visited_aisles:
                for item, qty in inst.aisles[a].items():
                    available[item] = available.get(item, 0) + qty

            deficits = {
                item: req - available.get(item, 0)
                for item, req in demand.items()
                if req > available.get(item, 0)
            }

            # If no explicit deficit was identified (e.g. no aisles returned),
            # remove the lowest-score order to shrink to an easier subset.
            if not deficits:
                to_remove = min(
                    current,
                    key=lambda o: (
                        order_sizes[o] - self.lambda_penalty * aisle_counts[o]
                    ),
                )
                current.remove(to_remove)
                continue

            to_remove = max(
                current,
                key=lambda o: (
                    sum(
                        min(inst.orders[o].get(item, 0), deficit)
                        for item, deficit in deficits.items()
                    ),
                    aisle_counts[o],
                    -order_sizes[o],
                ),
            )
            current.remove(to_remove)

        return [], []

    @staticmethod
    def _fractional_upper_bound(
        weights: list[int],
        values: list[float],
        ub: int,
        lb: int,
    ) -> float:
        """
        Fractional knapsack upper bound for the surrogate order-selection model.
        Returns -inf when even taking all candidate orders cannot reach lb.
        """
        if not weights:
            return float("-inf")

        total_weight = sum(weights)
        if total_weight < lb:
            return float("-inf")

        ranked = sorted(
            range(len(weights)),
            key=lambda i: (values[i] / weights[i]),
            reverse=True,
        )

        value_acc = 0.0
        weight_acc = 0

        for i in ranked:
            if weight_acc >= ub:
                break
            w_i = weights[i]
            v_i = values[i]

            if weight_acc + w_i <= ub:
                weight_acc += w_i
                value_acc += v_i
                continue

            remain = ub - weight_acc
            if remain > 0:
                frac = remain / w_i
                value_acc += frac * v_i
                weight_acc = ub
            break

        return value_acc if weight_acc >= lb else float("-inf")
