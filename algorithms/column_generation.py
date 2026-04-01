"""
Column Generation for aisle selection.

Treats each aisle as a "column" in a restricted master LP that minimises the
number of visited aisles while covering the demand of a greedily-selected
order set.

Algorithm outline:
  1. Select orders greedily (largest-first up to UB).
  2. Seed the restricted master with a small initial set of aisles (those
     with highest useful stock).
  3. Solve the restricted master LP  →  obtain dual prices (shadow prices)
     for the demand-covering constraints.
  4. Pricing step: scan all aisles NOT in the restricted master.  Aisle a
     has reduced cost  rc_a = 1 − Σ_i π_i · supply(a, i).  If rc_a < 0
     the aisle is profitable to add.
  5. Add the best negative-reduced-cost aisles and re-solve.  Repeat until
     no improving aisle exists or an iteration cap is reached.
  6. Round the LP solution: rank aisles by their LP values, sweep top-k
     subsets, re-pack orders, keep the best feasible configuration.

In the SBPO context |A| is typically small enough that direct LP methods
work well, so column generation mainly demonstrates the technique and can
help when many aisles are irrelevant.
"""

import numpy as np
from scipy.optimize import linprog

from algorithms.base import Algorithm
from problems.base import ProblemInput


class ColumnGeneration(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "column_generation"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders = inst.orders
        aisles = inst.aisles
        n_orders = inst.nOrders
        n_aisles = inst.nAisles
        lb = inst.lb
        ub = inst.ub

        max_cg_iterations = self.params.get("max_cg_iterations", 50)
        initial_pool_size = self.params.get("initial_pool_size", 5)
        max_columns_per_iter = self.params.get("max_columns_per_iter", 5)
        max_k = self.params.get("max_k", n_aisles)
        max_k = min(max_k, n_aisles)

        order_sizes = [sum(orders[i].values()) for i in range(n_orders)]

        # --- Step 1: Greedy order selection (largest-first) ---
        desc_indices = sorted(range(n_orders), key=lambda i: -order_sizes[i])
        initial_orders: list[int] = []
        total_units = 0
        for idx in desc_indices:
            if total_units + order_sizes[idx] <= ub:
                initial_orders.append(idx)
                total_units += order_sizes[idx]

        if total_units < lb:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        # Build demand vector from selected orders
        demand: dict[int, int] = {}
        for o in initial_orders:
            for item, qty in orders[o].items():
                demand[item] = demand.get(item, 0) + qty

        demanded_items = sorted(demand.keys())
        if not demanded_items:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        item_to_row = {item: i for i, item in enumerate(demanded_items)}
        n_demand_items = len(demanded_items)

        # --- Step 2: Seed the restricted master with high-useful-stock aisles ---
        useful_stocks = [
            (a, self._useful_stock(aisles[a], demand)) for a in range(n_aisles)
        ]
        useful_stocks.sort(key=lambda x: -x[1])
        pool = [a for a, _ in useful_stocks[:initial_pool_size]]
        pool_set = set(pool)

        # --- Step 3–5: Column generation loop ---
        for cg_iter in range(max_cg_iterations):
            # Solve restricted master LP
            ya, duals = self._solve_restricted_master(
                demand, demanded_items, item_to_row, n_demand_items,
                aisles, pool,
            )

            if duals is None:
                # LP infeasible with current pool; add more columns greedily
                for a, _ in useful_stocks:
                    if a not in pool_set:
                        pool.append(a)
                        pool_set.add(a)
                        if len(pool) >= min(n_aisles, len(pool) + max_columns_per_iter):
                            break
                continue

            # Pricing: find aisles with negative reduced cost
            candidates: list[tuple[float, int]] = []
            for a in range(n_aisles):
                if a in pool_set:
                    continue
                rc = 1.0
                for item, qty in aisles[a].items():
                    if item in item_to_row:
                        rc -= duals[item_to_row[item]] * qty
                if rc < -1e-8:
                    candidates.append((rc, a))

            if not candidates:
                break  # No improving columns — optimal for LP relaxation

            # Add best negative-reduced-cost columns
            candidates.sort()
            for rc, a in candidates[:max_columns_per_iter]:
                pool.append(a)
                pool_set.add(a)

        # --- Step 6: Final LP solve on full pool, then round ---
        ya_final, _ = self._solve_restricted_master(
            demand, demanded_items, item_to_row, n_demand_items,
            aisles, pool,
        )

        if ya_final is not None:
            # Map pool indices back to global aisle indices
            ranked_aisles = sorted(
                range(len(pool)),
                key=lambda i: (-ya_final[i], -self._useful_stock(aisles[pool[i]], demand)),
            )
            ranked_global = [pool[i] for i in ranked_aisles]
        else:
            # Fallback: rank pool aisles by useful stock
            ranked_global = sorted(
                pool, key=lambda a: -self._useful_stock(aisles[a], demand)
            )

        # Sweep top-k subsets and re-pack orders
        best_obj = -1.0
        best_result: dict = {}

        for k in range(1, min(max_k, len(ranked_global)) + 1):
            candidate_aisles = ranked_global[:k]
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

        if not best_result:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
        return best_result

    @staticmethod
    def _solve_restricted_master(
        demand: dict[int, int],
        demanded_items: list[int],
        item_to_row: dict[int, int],
        n_demand_items: int,
        aisles: list[dict[int, int]],
        pool: list[int],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Solve the restricted master LP:
            min  Σ_a ya
            s.t. Σ_a supply(a,i) · ya ≥ demand_i   for each demanded item i
                 0 ≤ ya ≤ 1

        Returns (ya, duals) where duals are the shadow prices π_i for the
        demand constraints, or (None, None) if infeasible.
        """
        n_pool = len(pool)
        if n_pool == 0:
            return None, None

        c = np.ones(n_pool)

        # -Σ supply(a,i) · ya ≤ -demand_i
        A_ub = np.zeros((n_demand_items, n_pool))
        b_ub = np.zeros(n_demand_items)

        for j, a in enumerate(pool):
            for item, qty in aisles[a].items():
                if item in item_to_row:
                    A_ub[item_to_row[item], j] = -qty

        for item, qty in demand.items():
            if item in item_to_row:
                b_ub[item_to_row[item]] = -qty

        bounds = [(0.0, 1.0)] * n_pool

        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
            options={"dual_feasibility_tolerance": 1e-7, "presolve": True},
        )

        if not result.success:
            return None, None

        # Extract dual variables (shadow prices for inequality constraints)
        # For HiGHS via linprog, the ineqlin dual is in result.ineqlin.marginals
        # These are the π_i values: the "price" of each demand constraint
        duals = None
        if hasattr(result, "ineqlin") and hasattr(result.ineqlin, "marginals"):
            # linprog returns non-positive duals for ≤ constraints;
            # negate to get positive shadow prices
            duals = -result.ineqlin.marginals
        return result.x, duals

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
