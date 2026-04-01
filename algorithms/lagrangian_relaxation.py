"""
Lagrangian relaxation on stock constraints for aisle selection.

Dualizes the stock-covering constraints  Σ_a supply(a,i)·y_a ≥ demand_i
with multipliers λ_i ≥ 0.  The relaxed aisle sub-problem decomposes per
aisle: aisle a is "free" when its Lagrangian reduced cost

    c_a = 1 − Σ_i λ_i · aisles[a][i]

is negative (the valued items it supplies outweigh the unit visit cost).

A subgradient loop tightens the multipliers; the resulting λ vector also
scores orders — orders whose items carry high multipliers are expensive to
cover, so we can re-select orders that avoid scarce items.  The outer loop
alternates between Lagrangian aisle selection and λ-guided order
re-selection, tracking the best feasible solution throughout.
"""

from algorithms.base import Algorithm
from problems.base import ProblemInput


class LagrangianRelaxation(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "lagrangian_relaxation"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders = inst.orders
        aisles = inst.aisles
        n_orders = inst.nOrders
        n_aisles = inst.nAisles
        lb = inst.lb
        ub = inst.ub

        max_outer = self.params.get("max_outer_iterations", 10)
        max_sg = self.params.get("max_subgradient_iterations", 100)
        step_init = self.params.get("step_size_init", 2.0)
        step_decay = self.params.get("step_decay", 0.95)

        order_sizes = [sum(o.values()) for o in orders]

        best_obj = -1.0
        best_result: dict = {}

        # Collect all items that appear in any order
        all_items: set[int] = set()
        for o in orders:
            all_items.update(o.keys())
        all_items_list = sorted(all_items)
        item_to_idx = {item: i for i, item in enumerate(all_items_list)}
        n_items = len(all_items_list)

        # Pre-compute aisle supply vectors (sparse, as dicts keyed by item_idx)
        aisle_supply: list[dict[int, int]] = []
        for a in range(n_aisles):
            supply = {}
            for item, qty in aisles[a].items():
                if item in item_to_idx:
                    supply[item_to_idx[item]] = qty
            aisle_supply.append(supply)

        # Pre-compute order demand vectors (sparse, keyed by item_idx)
        order_demand: list[dict[int, int]] = []
        for o in range(n_orders):
            dem = {}
            for item, qty in orders[o].items():
                dem[item_to_idx[item]] = qty
            order_demand.append(dem)

        # Initialize multipliers
        lam = [0.0] * n_items

        # --- Initial order selection: largest-first greedy ---
        sel_orders = self._greedy_order_select(order_sizes, n_orders, lb, ub)
        if not sel_orders:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        for outer in range(max_outer):
            demand = self._compute_demand(order_demand, sel_orders, n_items)

            # --- Subgradient loop for aisle selection ---
            step = step_init
            best_lagrangian_aisles: list[int] = []
            best_lagrangian_obj = -1.0
            last_sel_aisles: list[int] = []

            for sg_iter in range(max_sg):
                # Compute reduced costs and rank aisles
                reduced_costs: list[tuple[float, int]] = []
                for a in range(n_aisles):
                    rc = 1.0
                    for idx, qty in aisle_supply[a].items():
                        rc -= lam[idx] * qty
                    reduced_costs.append((rc, a))

                # Select aisles with negative reduced cost
                sel_aisles = [a for rc, a in reduced_costs if rc < 0]

                # If no aisle selected, pick the one with most negative cost
                if not sel_aisles:
                    reduced_costs.sort()
                    sel_aisles = [reduced_costs[0][1]]

                last_sel_aisles = sel_aisles

                # Compute supply from selected aisles
                supply = [0] * n_items
                for a in sel_aisles:
                    for idx, qty in aisle_supply[a].items():
                        supply[idx] += qty

                # Subgradient: g_i = demand_i - supply_i  (violation)
                g = [0.0] * n_items
                g_norm_sq = 0.0
                for i in range(n_items):
                    g[i] = demand[i] - supply[i]
                    g_norm_sq += g[i] * g[i]

                # Check if this aisle set yields a feasible + good solution
                if all(supply[i] >= demand[i] for i in range(n_items) if demand[i] > 0):
                    total_units = sum(order_sizes[o] for o in sel_orders)
                    if total_units >= lb and sel_aisles:
                        cleaned = self._cleanup_aisles(
                            orders, aisles, sel_orders, sel_aisles
                        )
                        if cleaned:
                            obj = total_units / len(cleaned)
                            if obj > best_lagrangian_obj:
                                best_lagrangian_obj = obj
                                best_lagrangian_aisles = cleaned

                # Update multipliers
                if g_norm_sq > 0:
                    for i in range(n_items):
                        lam[i] = max(0.0, lam[i] + step * g[i] / (g_norm_sq ** 0.5))

                step *= step_decay

            # Record best from this subgradient run
            if best_lagrangian_obj > best_obj:
                best_obj = best_lagrangian_obj
                best_result = {
                    "selected_orders": list(sel_orders),
                    "visited_aisles": list(best_lagrangian_aisles),
                    "objective": best_lagrangian_obj,
                }

            # Build candidate aisle sets to try re-packing against:
            # 1. Best feasible aisles from subgradient (if any)
            # 2. Last iteration's aisles (even if not feasible for original orders)
            # 3. Sweep top-k aisles ranked by reduced cost (like LP approach)
            reduced_costs_final = []
            for a in range(n_aisles):
                rc = 1.0
                for idx, qty in aisle_supply[a].items():
                    rc -= lam[idx] * qty
                reduced_costs_final.append((rc, a))
            reduced_costs_final.sort()
            ranked_aisles = [a for _, a in reduced_costs_final]

            candidate_aisle_sets: list[list[int]] = []
            if best_lagrangian_aisles:
                candidate_aisle_sets.append(best_lagrangian_aisles)
            if last_sel_aisles:
                candidate_aisle_sets.append(last_sel_aisles)
            # Add top-k sweeps
            max_k_sweep = min(self.params.get("max_k_sweep", 30), n_aisles)
            for k in range(1, max_k_sweep + 1):
                candidate_aisle_sets.append(ranked_aisles[:k])

            for aisle_set in candidate_aisle_sets:
                for sort_key in [
                    lambda o: -order_sizes[o],
                    lambda o: order_sizes[o],
                    lambda o: self._order_lagrangian_cost(order_demand[o], lam),
                    lambda o: -self._order_lagrangian_cost(order_demand[o], lam),
                ]:
                    repacked, total = self._pack_orders_with_inventory(
                        orders, order_sizes, aisles, aisle_set,
                        sorted(range(n_orders), key=sort_key), ub,
                    )
                    if total >= lb and repacked:
                        cleaned = self._cleanup_aisles(
                            orders, aisles, repacked, aisle_set
                        )
                        if cleaned:
                            obj = total / len(cleaned)
                            if obj > best_obj:
                                best_obj = obj
                                best_result = {
                                    "selected_orders": list(repacked),
                                    "visited_aisles": list(cleaned),
                                    "objective": obj,
                                }

            # --- Re-select orders guided by λ ---
            # Orders with low Lagrangian cost are cheaper to cover
            sel_orders = self._lambda_guided_order_select(
                order_demand, order_sizes, lam, n_orders, lb, ub
            )
            if not sel_orders:
                # Fall back to greedy
                sel_orders = self._greedy_order_select(order_sizes, n_orders, lb, ub)
                if not sel_orders:
                    break

        if not best_result:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
        return best_result

    @staticmethod
    def _greedy_order_select(
        order_sizes: list[int], n_orders: int, lb: int, ub: int
    ) -> list[int]:
        """Select orders largest-first up to ub."""
        desc = sorted(range(n_orders), key=lambda i: -order_sizes[i])
        selected: list[int] = []
        total = 0
        for idx in desc:
            if total + order_sizes[idx] <= ub:
                selected.append(idx)
                total += order_sizes[idx]
        if total < lb:
            return []
        return selected

    @staticmethod
    def _lambda_guided_order_select(
        order_demand: list[dict[int, int]],
        order_sizes: list[int],
        lam: list[float],
        n_orders: int,
        lb: int,
        ub: int,
    ) -> list[int]:
        """Select orders preferring those with low Lagrangian cost per unit."""
        # cost_per_unit = Σ_i λ_i * demand_i(o) / size(o)
        costs = []
        for o in range(n_orders):
            size = order_sizes[o]
            if size == 0:
                continue
            cost = sum(lam[idx] * qty for idx, qty in order_demand[o].items())
            costs.append((cost / size, o))

        # Sort by cost/unit ascending (cheapest to cover first)
        costs.sort()

        selected: list[int] = []
        total = 0
        for _, idx in costs:
            if total + order_sizes[idx] <= ub:
                selected.append(idx)
                total += order_sizes[idx]
        if total < lb:
            return []
        return selected

    @staticmethod
    def _compute_demand(
        order_demand: list[dict[int, int]], sel_orders: list[int], n_items: int
    ) -> list[int]:
        demand = [0] * n_items
        for o in sel_orders:
            for idx, qty in order_demand[o].items():
                demand[idx] += qty
        return demand

    @staticmethod
    def _order_lagrangian_cost(
        order_dem: dict[int, int], lam: list[float]
    ) -> float:
        return sum(lam[idx] * qty for idx, qty in order_dem.items())

    @staticmethod
    def _pack_orders_with_inventory(
        orders: list[dict[int, int]],
        order_sizes: list[int],
        aisles: list[dict[int, int]],
        aisle_indices: list[int],
        sequence: list[int],
        ub: int,
    ) -> tuple[list[int], int]:
        """Pack orders in sequence respecting pooled inventory and ub."""
        inventory: dict[int, int] = {}
        for a in aisle_indices:
            for item, qty in aisles[a].items():
                inventory[item] = inventory.get(item, 0) + qty
        remaining = dict(inventory)

        selected: list[int] = []
        total = 0
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
