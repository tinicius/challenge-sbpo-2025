"""
Approach A — Seed-Order Knapsack.

Picks the largest order as a "seed", identifies which aisles hold its items,
then scores every other order by how much it overlaps with those seed aisles:

    value(o) = total_items(o) × overlap_fraction(o)

where overlap_fraction(o) = |items(o) ∩ items_in_seed_aisles| / |items(o)|.

Running the Knapsack with these values biases selection toward orders that
share aisles with the seed, yielding a tightly clustered wave that the
downstream greedy aisle selection can cover with fewer aisles.
"""

from algorithms.base import Algorithm
from algorithms.utils.knapsack_dp import knapsack_dp, repair_infeasible
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class KnapsackSeedSolver(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "knapsack_seed"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders, aisles = inst.orders, inst.aisles
        n_orders, n_aisles = inst.nOrders, inst.nAisles
        lb, ub = inst.lb, inst.ub

        order_sizes = [sum(o.values()) for o in orders]

        # Pick seed: largest order by total units
        seed_idx = max(range(n_orders), key=lambda i: order_sizes[i])
        seed_items = set(orders[seed_idx].keys())

        # Collect all item types available in aisles that carry any seed item
        seed_aisle_item_set: set[int] = set()
        for a in range(n_aisles):
            if seed_items & aisles[a].keys():
                seed_aisle_item_set.update(aisles[a].keys())

        # Value = size × overlap_fraction
        values: list[float] = []
        for i, order in enumerate(orders):
            items_in_order = set(order.keys())
            if not items_in_order:
                values.append(0.0)
                continue
            overlap = len(items_in_order & seed_aisle_item_set) / len(items_in_order)
            values.append(order_sizes[i] * overlap)

        weights = order_sizes
        selected_orders = knapsack_dp(weights, values, ub, lb)

        if selected_orders is None:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

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
