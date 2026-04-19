"""
Dinkelbach Method solver for the wave order-picking problem.

Based on:
  "Optimal Order Selection via the Dinkelbach Method"
  Leal, Castro, Longo — SBPO 2025

Uses HiGHS MIP solver with:
- LP-based branch-and-bound
- Model reuse across iterations (only objective coefficients change)
- True MIP-start warm-starting
"""

import time

import numpy as np
import highspy

from algorithms.base import Algorithm
from problems.base import ProblemInput


class DinkelbachMIP(Algorithm):

    @property
    def name(self) -> str:
        return "dinkelbach_mip"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        time_limit: float = float(self.params.get("time_limit", 590))
        max_iters: int = int(self.params.get("max_iters", 15))
        epsilon: float = float(self.params.get("epsilon", 1e-3))
        num_threads: int = int(self.params.get("num_threads", 0))

        start_time = time.time()

        order_sizes = [sum(o.values()) for o in inst.orders]

        demanded_items: set[int] = set()
        for o in inst.orders:
            demanded_items.update(o.keys())

        active_aisles = [
            a
            for a in range(inst.nAisles)
            if any(item in demanded_items for item in inst.aisles[a])
        ]

        # --- Build MIP model once ---
        solver, n_cols, aisle_col = self._build_model(
            inst, order_sizes, demanded_items, active_aisles, num_threads
        )

        # --- Warm-start: seed-expansion heuristic ---
        best_sol = self._warmstart(inst, order_sizes)

        if best_sol["objective"] > 0.0:
            items_ws = sum(order_sizes[o] for o in best_sol["selected_orders"])
            aisles_ws = len(best_sol["visited_aisles"])
            alpha = items_ws / aisles_ws
        else:
            alpha = 0.0

        for k in range(max_iters):
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            if remaining < 1.0:
                break

            # Time budget: ensure room for multiple iterations.
            # Use at most remaining/3 so at least 3 iterations can run,
            # but give all remaining time on the last iteration.
            iters_left = max_iters - k
            if iters_left <= 1:
                iter_budget = remaining
            else:
                iter_budget = min(remaining, remaining / 3)
            solver.setOptionValue("time_limit", iter_budget)

            # Update objective coefficients only (paper Section 7):
            #   max Σ s_o * y_o − α * Σ x_a
            for o in range(inst.nOrders):
                solver.changeColCost(o, order_sizes[o])
            for a in active_aisles:
                solver.changeColCost(aisle_col[a], -alpha)

            # MIP-start from best known solution
            if best_sol["objective"] > 0.0:
                self._inject_mip_start(
                    solver, n_cols, inst.nOrders, aisle_col, best_sol
                )

            solver.solve()
            model_status = solver.getModelStatus()

            feasible_statuses = {
                highspy.HighsModelStatus.kOptimal,
                highspy.HighsModelStatus.kObjectiveBound,
                highspy.HighsModelStatus.kObjectiveTarget,
                highspy.HighsModelStatus.kSolutionLimit,
                highspy.HighsModelStatus.kTimeLimit,
            }

            if model_status not in feasible_statuses:
                prim_status = solver.getInfoValue("primal_solution_status")[1]
                if prim_status != 2:  # 2 = feasible
                    break

            was_optimal = model_status == highspy.HighsModelStatus.kOptimal

            sol = solver.getSolution()
            col_vals = sol.col_value

            selected_orders = [
                o for o in range(inst.nOrders) if col_vals[o] > 0.5
            ]
            visited_aisles = [
                a for a in active_aisles if col_vals[aisle_col[a]] > 0.5
            ]

            if not visited_aisles:
                break

            items = sum(order_sizes[o] for o in selected_orders)
            if items < inst.lb:
                break

            n_aisles = len(visited_aisles)
            ratio = items / n_aisles

            if ratio > best_sol["objective"]:
                best_sol = {
                    "selected_orders": selected_orders,
                    "visited_aisles": visited_aisles,
                    "objective": ratio,
                }

            # Dinkelbach convergence (paper Section 5):
            # Only trust convergence when MIP was solved to optimality.
            residual = abs(items - alpha * n_aisles)
            if was_optimal and residual < epsilon:
                break
            if was_optimal and ratio <= alpha + 1e-9:
                break

            # For non-optimal solves: still update α if we improved
            if ratio > alpha:
                alpha = ratio

        return best_sol

    # ------------------------------------------------------------------ #
    #  MIP model construction                                             #
    # ------------------------------------------------------------------ #

    def _build_model(
        self,
        inst: ProblemInput,
        order_sizes: list[int],
        demanded_items: set[int],
        active_aisles: list[int],
        num_threads: int,
    ) -> tuple[highspy.Highs, int, dict[int, int]]:
        """Build the MIP model once. Returns (solver, n_cols, aisle_col_map)."""
        solver = highspy.Highs()
        solver.silent()
        solver.changeObjectiveSense(highspy.ObjSense.kMaximize)
        if num_threads > 0:
            solver.setOptionValue("threads", num_threads)

        n_orders = inst.nOrders
        orders = inst.orders
        aisles = inst.aisles

        # Columns 0..n_orders-1: y_o (order selection)
        for o in range(n_orders):
            solver.addBinary(order_sizes[o], f"y_{o}")

        # Columns n_orders..: x_a (aisle selection)
        aisle_col: dict[int, int] = {}
        for i, a in enumerate(active_aisles):
            aisle_col[a] = n_orders + i
            solver.addBinary(0.0, f"x_{a}")

        n_cols = n_orders + len(active_aisles)

        # Constraint (2): LB ≤ Σ s_o * y_o ≤ UB
        order_indices = list(range(n_orders))
        order_coeffs = [float(order_sizes[o]) for o in range(n_orders)]
        solver.addRow(
            float(inst.lb), float(inst.ub),
            len(order_indices),
            np.array(order_indices, dtype=np.int32),
            np.array(order_coeffs, dtype=np.float64),
        )

        # Constraint (3): Σ_o u_oi * y_o ≤ Σ_a u_ai * x_a   ∀i
        for item in demanded_items:
            row_indices = []
            row_coeffs = []

            for o in range(n_orders):
                if item in orders[o]:
                    row_indices.append(o)
                    row_coeffs.append(float(orders[o][item]))

            if not row_indices:
                continue

            for a in active_aisles:
                if item in aisles[a]:
                    row_indices.append(aisle_col[a])
                    row_coeffs.append(-float(aisles[a][item]))

            solver.addRow(
                -highspy.kHighsInf, 0.0,
                len(row_indices),
                np.array(row_indices, dtype=np.int32),
                np.array(row_coeffs, dtype=np.float64),
            )

        return solver, n_cols, aisle_col

    def _inject_mip_start(
        self,
        solver: highspy.Highs,
        n_cols: int,
        n_orders: int,
        aisle_col: dict[int, int],
        sol: dict,
    ) -> None:
        """Inject a feasible solution as MIP-start."""
        hint_orders = set(sol["selected_orders"])
        hint_aisles = set(sol["visited_aisles"])

        indices = []
        values = []
        for o in range(n_orders):
            indices.append(o)
            values.append(1.0 if o in hint_orders else 0.0)
        for a, col in aisle_col.items():
            indices.append(col)
            values.append(1.0 if a in hint_aisles else 0.0)

        solver.setSolution(
            len(indices),
            np.array(indices, dtype=np.int32),
            np.array(values, dtype=np.float64),
        )

    # ------------------------------------------------------------------ #
    #  Warm-start heuristic: aisle-seed expansion                         #
    # ------------------------------------------------------------------ #
    #                                                                      #
    #  For each of the top-K aisles (ranked by useful inventory), try it   #
    #  as the sole seed aisle and greedily pack orders that fit within its  #
    #  stock.  If the wave lower bound isn't met, expand by adding the     #
    #  most similar aisle (Jaccard on item sets) and retry.  After packing #
    #  orders, prune redundant aisles to shrink the denominator.           #
    #                                                                      #
    #  Multiple order-selection strategies (smallest-first, largest-first, #
    #  fewest-item-types, highest-density) are tried for each seed set.    #
    #  The best ratio across all (seed, strategy) combos is returned.      #
    # ------------------------------------------------------------------ #

    def _warmstart(self, inst: ProblemInput, order_sizes: list[int]) -> dict:
        orders = inst.orders
        aisles = inst.aisles
        n_orders = inst.nOrders
        n_aisles = inst.nAisles
        lb = inst.lb
        ub = inst.ub

        # Pre-compute total demand (for aisle scoring)
        total_demand: dict[int, int] = {}
        for o in orders:
            for item, qty in o.items():
                total_demand[item] = total_demand.get(item, 0) + qty

        # Rank aisles by useful inventory (min of stock and demand per item)
        aisle_scores: list[tuple[int, int]] = []
        for a in range(n_aisles):
            useful = sum(
                min(qty, total_demand.get(item, 0))
                for item, qty in aisles[a].items()
            )
            aisle_scores.append((a, useful))
        aisle_scores.sort(key=lambda x: -x[1])
        ranked_aisles = [a for a, _ in aisle_scores]

        # Aisle item sets for Jaccard similarity
        aisle_items = [set(aisles[a].keys()) for a in range(n_aisles)]

        # Order-selection strategies
        order_by_size_asc = sorted(range(n_orders), key=lambda o: order_sizes[o])
        order_by_size_desc = sorted(
            range(n_orders), key=lambda o: -order_sizes[o]
        )
        order_by_items_asc = sorted(
            range(n_orders), key=lambda o: len(orders[o])
        )
        order_by_density = sorted(
            range(n_orders),
            key=lambda o: -order_sizes[o] / max(1, len(orders[o])),
        )
        strategies = [
            order_by_size_asc,
            order_by_size_desc,
            order_by_items_asc,
            order_by_density,
        ]

        max_seeds = min(10, len(ranked_aisles))

        best: dict = {
            "selected_orders": [],
            "visited_aisles": [],
            "objective": 0.0,
        }

        for seed_aisle in ranked_aisles[:max_seeds]:
            # Sort other aisles by Jaccard similarity to the seed
            seed_items = aisle_items[seed_aisle]
            others = [a for a in ranked_aisles if a != seed_aisle]
            others.sort(
                key=lambda a: self._jaccard(seed_items, aisle_items[a]),
                reverse=True,
            )

            # Expand from 1 aisle until we can fill at least LB items
            chosen_aisles = [seed_aisle]
            expand_idx = 0

            while True:
                inventory = self._pool_inventory(aisles, chosen_aisles)

                for seq in strategies:
                    sel, units = self._pack_orders(
                        orders, order_sizes, inventory, seq, lb, ub
                    )
                    if units < lb:
                        continue

                    demand = self._compute_demand(orders, sel)
                    pruned = self._prune_aisles(demand, aisles, list(chosen_aisles))
                    obj = units / len(pruned)

                    if obj > best["objective"]:
                        best = {
                            "selected_orders": sel,
                            "visited_aisles": pruned,
                            "objective": obj,
                        }

                # Try expanding to next most-similar aisle
                if expand_idx < len(others):
                    chosen_aisles.append(others[expand_idx])
                    expand_idx += 1
                    # Check if any previous strategy already met LB with
                    # fewer aisles — if so, expanding further can only hurt
                    # the ratio unless we couldn't meet LB yet.
                    inv = self._pool_inventory(aisles, chosen_aisles)
                    max_packable = sum(
                        order_sizes[o] for o in range(n_orders)
                        if all(
                            inv.get(item, 0) >= qty
                            for item, qty in orders[o].items()
                        )
                    )
                    if max_packable < lb:
                        continue  # keep expanding, we need more stock
                    # If current best uses fewer aisles and met LB, stop expanding
                    if best["objective"] > 0 and len(chosen_aisles) > 2 * len(
                        best["visited_aisles"]
                    ):
                        break
                else:
                    break

        return best

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b)
        return inter / (len(a) + len(b) - inter)

    @staticmethod
    def _pool_inventory(
        aisles: list[dict[int, int]], indices: list[int]
    ) -> dict[int, int]:
        inv: dict[int, int] = {}
        for a in indices:
            for item, qty in aisles[a].items():
                inv[item] = inv.get(item, 0) + qty
        return inv

    @staticmethod
    def _pack_orders(
        orders: list[dict[int, int]],
        order_sizes: list[int],
        inventory: dict[int, int],
        sequence: list[int],
        lb: int,
        ub: int,
    ) -> tuple[list[int], int]:
        remaining = dict(inventory)
        selected: list[int] = []
        total = 0

        for o in sequence:
            units = order_sizes[o]
            if total + units > ub:
                continue
            feasible = all(
                remaining.get(item, 0) >= qty
                for item, qty in orders[o].items()
            )
            if not feasible:
                continue
            selected.append(o)
            total += units
            for item, qty in orders[o].items():
                remaining[item] -= qty

        return selected, total

    @staticmethod
    def _compute_demand(
        orders: list[dict[int, int]], selected: list[int]
    ) -> dict[int, int]:
        demand: dict[int, int] = {}
        for o in selected:
            for item, qty in orders[o].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    @staticmethod
    def _prune_aisles(
        original_demand: dict[int, int],
        aisles: list[dict[int, int]],
        visited: list[int],
    ) -> list[int]:
        """Remove aisles whose supply is redundant."""
        demand = dict(original_demand)
        current = list(visited)

        scored = []
        for a in current:
            contrib = sum(
                min(qty, demand.get(item, 0))
                for item, qty in aisles[a].items()
            )
            scored.append((a, contrib))
        scored.sort(key=lambda x: x[1])

        for a, _ in scored:
            if len(current) <= 1:
                break
            candidate = [idx for idx in current if idx != a]
            supply: dict[int, int] = {}
            for idx in candidate:
                for item, qty in aisles[idx].items():
                    supply[item] = supply.get(item, 0) + qty
            if all(supply.get(item, 0) >= qty for item, qty in demand.items()):
                current = candidate

        return current
