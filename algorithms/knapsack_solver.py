from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class KnapsackSolver(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "knapsack"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders, aisles = inst.orders, inst.aisles
        lb, ub = inst.lb, inst.ub

        order_sizes = [sum(o.values()) for o in orders]

        # Phase 1: Knapsack DP — select orders maximizing total units in [lb, ub]
        selected_orders = self._knapsack_dp(order_sizes, ub, lb)

        if selected_orders is None:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        # Phase 1.5: Repair — remove orders whose items can't be sourced from any aisle
        selected_orders, demand = self._repair_infeasible(
            selected_orders, orders, aisles, order_sizes, lb
        )
        if selected_orders is None:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        total_units = sum(order_sizes[o] for o in selected_orders)

        # Phase 2: Greedily select minimum aisles (demand already built by repair step)
        visited_aisles = multi_greedy_aisle_select(demand, aisles)

        if not visited_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        obj = total_units / len(visited_aisles)
        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": obj,
        }

    @staticmethod
    def _repair_infeasible(
        selected_orders: list[int],
        orders: list[dict],
        aisles: list[dict],
        order_sizes: list[int],
        lb: int,
    ) -> tuple[list[int], dict[int, int]] | tuple[None, None]:
        """
        Remove orders until all demanded items can be satisfied by the full aisle pool.
        Returns (repaired_orders, demand) or (None, None) if total units would drop below lb.
        """
        total_available: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                total_available[item] = total_available.get(item, 0) + qty

        selected = list(selected_orders)

        while selected:
            demand: dict[int, int] = {}
            for o in selected:
                for item, qty in orders[o].items():
                    demand[item] = demand.get(item, 0) + qty

            uncoverable = {
                item for item, qty in demand.items()
                if total_available.get(item, 0) < qty
            }

            if not uncoverable:
                return selected, demand

            order_to_remove = max(
                selected,
                key=lambda o: (
                    sum(1 for item in orders[o] if item in uncoverable),
                    -order_sizes[o],
                ),
            )
            selected.remove(order_to_remove)

            if sum(order_sizes[o] for o in selected) < lb:
                return None, None

        return None, None

    @staticmethod
    def _knapsack_dp(weights: list[int], ub: int, lb: int) -> list[int] | None:
        """
        0/1 Knapsack DP where value = weight for all items.
        Returns a list of order indices achieving max total weight in [lb, ub],
        or None if no feasible subset exists.

        Falls back to greedy descending fill when n * ub > 5M (memory guard).
        """
        n = len(weights)
        if n == 0:
            return None

        MAX_CELLS = 5_000_000
        if n * ub > MAX_CELLS:
            # Greedy fallback: pack largest orders first
            order = sorted(range(n), key=lambda i: -weights[i])
            selected, total = [], 0
            for i in order:
                if total + weights[i] <= ub:
                    selected.append(i)
                    total += weights[i]
            return selected if total >= lb else None

        # dp[w] = True if weight w is reachable using some subset of orders seen so far
        # keep[i][w] = True if order i was the last one included to reach weight w
        dp = [False] * (ub + 1)
        dp[0] = True
        keep = [[False] * (ub + 1) for _ in range(n)]

        for i, w_i in enumerate(weights):
            for w in range(ub, w_i - 1, -1):
                if dp[w - w_i] and not dp[w]:
                    dp[w] = True
                    keep[i][w] = True

        # Find max reachable weight in [lb, ub]
        best_w = next((w for w in range(ub, lb - 1, -1) if dp[w]), None)
        if best_w is None:
            return None

        # Backtrack to recover selected order indices
        selected, w = [], best_w
        for i in range(n - 1, -1, -1):
            if keep[i][w]:
                selected.append(i)
                w -= weights[i]
        return selected
