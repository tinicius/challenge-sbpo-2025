"""
Approach B — Density-Scored Knapsack.

Instead of treating all items equally, score each order by its pick density:

    value(o) = total_items(o) / A(o)

where A(o) is the minimum number of aisles needed to fulfil order o alone
(computed via greedy set cover on the single-order demand).

Running the Knapsack with these values favours orders that are naturally
concentrated in few aisles, so the combined wave needs fewer aisles and
yields a better items/aisles objective.
"""

from algorithms.base import Algorithm
from algorithms.utils.knapsack_dp import knapsack_dp, repair_infeasible
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class KnapsackDensitySolver(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "knapsack_density"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders, aisles = inst.orders, inst.aisles
        n_orders = inst.nOrders
        lb, ub = inst.lb, inst.ub

        order_sizes = [sum(o.values()) for o in orders]

        # Pre-compute A(o): minimum aisles needed for each order individually
        aisle_counts: list[int] = []
        for order in orders:
            if not order:
                aisle_counts.append(0)
                continue
            needed = multi_greedy_aisle_select(dict(order), aisles)
            aisle_counts.append(len(needed))

        # Value = total_items / A(o); skip unfulfillable orders (A(o) == 0)
        values: list[float] = []
        weights: list[int] = []
        order_indices: list[int] = []  # maps filtered positions back to original indices

        for i in range(n_orders):
            if aisle_counts[i] == 0 or order_sizes[i] == 0:
                continue
            values.append(order_sizes[i] / aisle_counts[i])
            weights.append(order_sizes[i])
            order_indices.append(i)

        if not order_indices:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        positions = knapsack_dp(weights, values, ub, lb)

        if positions is None:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        selected_orders = [order_indices[p] for p in positions]
        selected_orders = repair_infeasible(orders, order_sizes, selected_orders, aisles, lb)
        if not selected_orders:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        total_units = sum(order_sizes[o] for o in selected_orders)

        demand: dict[int, int] = {}
        for o in selected_orders:
            for item, qty in orders[o].items():
                demand[item] = demand.get(item, 0) + qty

        visited_aisles = multi_greedy_aisle_select(demand, aisles)

        if not visited_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        obj = total_units / len(visited_aisles)
        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": obj,
        }
