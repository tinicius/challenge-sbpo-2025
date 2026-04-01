"""
Aisle-focus heuristic via LP relaxation + rounding.

Solves a set-cover-style LP relaxation where continuous variables ya ∈ [0,1]
indicate whether aisle a is visited. The LP minimises the number of aisles
needed to cover the demand of a greedily-selected set of orders.

The LP solution yields:
  1. A lower bound on |A'| (quality certificate).
  2. A ranking of aisles by their fractional LP values.

We then sweep over top-k aisle subsets (ordered by LP value), re-pack orders
against each subset's inventory, clean up redundant aisles, and keep the
configuration with the best items/aisles objective.
"""

import numpy as np
from scipy.optimize import linprog

from algorithms.base import Algorithm
from problems.base import ProblemInput


class LPAisleFocus(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "lp_aisle_focus"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders = inst.orders
        aisles = inst.aisles
        n_orders = inst.nOrders
        n_aisles = inst.nAisles
        lb = inst.lb
        ub = inst.ub

        max_k = self.params.get("max_k", n_aisles)
        max_k = min(max_k, n_aisles)

        # Pre-compute order sizes
        order_sizes = [sum(orders[i].values()) for i in range(n_orders)]

        # --- Step 1: Build demand from largest-first order selection ---
        desc_indices = sorted(range(n_orders), key=lambda i: -order_sizes[i])
        initial_orders: list[int] = []
        total_units = 0
        for idx in desc_indices:
            if total_units + order_sizes[idx] <= ub:
                initial_orders.append(idx)
                total_units += order_sizes[idx]

        if total_units < lb:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        demand: dict[int, int] = {}
        for o in initial_orders:
            for item, qty in orders[o].items():
                demand[item] = demand.get(item, 0) + qty

        demanded_items = sorted(demand.keys())
        if not demanded_items:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        # --- Step 2: Solve LP to get aisle ranking ---
        ya = self._solve_lp(demand, demanded_items, aisles, n_aisles)

        if ya is not None:
            # Rank aisles by LP value (descending), break ties by useful stock
            ranked_aisles = sorted(
                range(n_aisles),
                key=lambda a: (-ya[a], -self._useful_stock(aisles[a], demand)),
            )
            lp_lb = float(np.sum(ya))
        else:
            # LP infeasible: fall back to greedy useful-stock ranking
            ranked_aisles = sorted(
                range(n_aisles),
                key=lambda a: -self._useful_stock(aisles[a], demand),
            )
            lp_lb = None

        # --- Step 3: Sweep top-k aisle subsets, re-pack orders, keep best ---
        best_obj = -1.0
        best_result: dict = {}

        for k in range(1, max_k + 1):
            candidate_aisles = ranked_aisles[:k]
            inventory = self._pool_inventory(aisles, candidate_aisles)

            for sort_key in [
                lambda i: -order_sizes[i],
                lambda i: order_sizes[i],
            ]:
                sequence = sorted(range(n_orders), key=sort_key)
                sel_orders, sel_total = self._pack_orders(
                    orders, order_sizes, inventory, sequence, ub
                )
                if sel_total < lb:
                    continue

                final_aisles = self._cleanup_aisles(
                    orders, aisles, sel_orders, candidate_aisles
                )
                if not final_aisles:
                    continue

                obj = sel_total / len(final_aisles)
                if obj > best_obj:
                    best_obj = obj
                    best_result = {
                        "selected_orders": sel_orders,
                        "visited_aisles": final_aisles,
                        "objective": obj,
                    }

        if lp_lb is not None and best_result:
            best_result["lp_lower_bound_aisles"] = lp_lb

        if not best_result:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
        return best_result

    @staticmethod
    def _solve_lp(
        demand: dict[int, int],
        demanded_items: list[int],
        aisles: list[dict[int, int]],
        n_aisles: int,
    ) -> np.ndarray | None:
        """Solve min Σ ya s.t. supply covers demand. Returns ya or None."""
        item_to_row = {item: i for i, item in enumerate(demanded_items)}
        n_items = len(demanded_items)

        c = np.ones(n_aisles)
        A_ub = np.zeros((n_items, n_aisles))
        b_ub = np.zeros(n_items)

        for a in range(n_aisles):
            for item, qty in aisles[a].items():
                if item in item_to_row:
                    A_ub[item_to_row[item], a] = -qty

        for item, qty in demand.items():
            b_ub[item_to_row[item]] = -qty

        bounds = [(0.0, 1.0)] * n_aisles
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            return result.x
        return None

    @staticmethod
    def _useful_stock(aisle: dict[int, int], demand: dict[int, int]) -> int:
        return sum(min(qty, demand.get(item, 0)) for item, qty in aisle.items())

    @staticmethod
    def _pool_inventory(
        aisles: list[dict[int, int]], aisle_indices: list[int]
    ) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for a in aisle_indices:
            for item, qty in aisles[a].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    @staticmethod
    def _pack_orders(
        orders: list[dict[int, int]],
        order_sizes: list[int],
        inventory: dict[int, int],
        sequence: list[int],
        ub: int,
    ) -> tuple[list[int], int]:
        """Pack orders in given sequence respecting inventory and UB."""
        selected: list[int] = []
        total = 0
        remaining = dict(inventory)

        for idx in sequence:
            size = order_sizes[idx]
            if total + size > ub:
                continue
            order = orders[idx]
            if all(remaining.get(item, 0) >= qty for item, qty in order.items()):
                selected.append(idx)
                total += size
                for item, qty in order.items():
                    remaining[item] -= qty

        return selected, total

    @staticmethod
    def _cleanup_aisles(
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        selected_orders: list[int],
        aisle_indices: list[int],
    ) -> list[int]:
        """Remove aisles whose removal doesn't break feasibility."""
        demand: dict[int, int] = {}
        for o in selected_orders:
            for item, qty in orders[o].items():
                demand[item] = demand.get(item, 0) + qty

        current = list(aisle_indices)
        scored = []
        for a in current:
            contrib = sum(
                min(qty, demand.get(item, 0)) for item, qty in aisles[a].items()
            )
            scored.append((a, contrib))
        scored.sort(key=lambda x: (x[1], x[0]))

        for a, _ in scored:
            if len(current) <= 1:
                break
            candidate = [x for x in current if x != a]
            supply: dict[int, int] = {}
            for ca in candidate:
                for item, qty in aisles[ca].items():
                    supply[item] = supply.get(item, 0) + qty
            if all(supply.get(item, 0) >= qty for item, qty in demand.items()):
                current = candidate

        return current
