import random

from models.solver import Solver


class AisleFirstHeuristic(Solver):

    def _build_total_demand(self) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order in self.orders:
            for item, qty in order.items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _score_aisles(self, total_demand: dict[int, int]) -> list[tuple[int, int]]:
        scored: list[tuple[int, int]] = []
        for aisle_idx, aisle in enumerate(self.aisles):
            useful_inventory = 0
            for item, qty in aisle.items():
                useful_inventory += min(qty, total_demand.get(item, 0))
            scored.append((aisle_idx, useful_inventory))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def _pool_inventory(self, aisle_indices: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _is_order_feasible_against_inventory(
        self, order: dict[int, int], inventory: dict[int, int]
    ) -> bool:
        for item, qty in order.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _build_order_sequences(self) -> list[list[int]]:
        strategy = self.config.get("order_strategy", "both")
        base = self.config.get("seed", list(range(self.n_orders)))
        base_list = list(base)

        if strategy == "seed":
            return [base_list]

        asc = sorted(base_list, key=lambda idx: (sum(self.orders[idx].values()), idx))
        desc = sorted(base_list, key=lambda idx: (-sum(self.orders[idx].values()), idx))

        if strategy == "size_asc":
            return [asc]

        if strategy == "size_desc":
            return [desc]

        return [desc, asc]

    def _build_demand_from_orders(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _demand_is_coverable_by_aisles(
        self, demand: dict[int, int], aisle_indices: list[int]
    ) -> bool:
        if not demand:
            return True

        inventory = self._pool_inventory(aisle_indices)
        for item, qty in demand.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _cleanup_redundant_aisles(
        self, selected_orders: list[int], aisle_indices: list[int]
    ) -> list[int]:
        if not selected_orders or not aisle_indices:
            return []

        demand = self._build_demand_from_orders(selected_orders)
        current_aisles = sorted(set(aisle_indices))

        # Remove low-contribution aisles first to shrink denominator faster.
        scored_aisles: list[tuple[int, int]] = []
        for aisle_idx in current_aisles:
            contribution = sum(
                min(qty, demand.get(item, 0))
                for item, qty in self.aisles[aisle_idx].items()
            )
            scored_aisles.append((aisle_idx, contribution))

        for aisle_idx, _ in sorted(scored_aisles, key=lambda x: (x[1], x[0])):
            if len(current_aisles) == 1:
                break

            candidate_aisles = [idx for idx in current_aisles if idx != aisle_idx]
            if self._demand_is_coverable_by_aisles(demand, candidate_aisles):
                current_aisles = candidate_aisles

        return current_aisles

    def _choose_aisles_for_k(
        self,
        ranked_aisles: list[int],
        k: int,
        rcl_extra: int,
        rng: random.Random,
    ) -> list[int]:
        if rcl_extra <= 0:
            return ranked_aisles[:k]

        rcl_size = min(len(ranked_aisles), k + rcl_extra)
        rcl = ranked_aisles[:rcl_size]
        if len(rcl) <= k:
            return list(rcl)

        sampled = rng.sample(rcl, k)
        sampled.sort()
        return sampled

    def _pack_orders_for_inventory(
        self, inventory_pool: dict[int, int], order_sequence: list[int]
    ) -> tuple[list[int], int]:
        selected_orders: list[int] = []
        total_units = 0

        remaining_inventory = dict(inventory_pool)

        for order_idx in order_sequence:
            order = self.orders[order_idx]
            order_units = sum(order.values())

            if total_units + order_units > self.ub:
                continue

            if not self._is_order_feasible_against_inventory(
                order, remaining_inventory
            ):
                continue

            selected_orders.append(order_idx)
            total_units += order_units

            for item, qty in order.items():
                remaining_inventory[item] = remaining_inventory.get(item, 0) - qty

        return selected_orders, total_units

    def solve(self) -> tuple[list[int], list[int]]:
        total_demand = self._build_total_demand()
        ranked_scores = self._score_aisles(total_demand)
        ranked_aisles = [aisle_idx for aisle_idx, _ in ranked_scores]

        if not ranked_aisles:
            return [], []

        max_k_cfg = self.config.get("max_k", self.n_aisles)
        max_k = max(1, min(max_k_cfg, self.n_aisles))

        order_sequences = self._build_order_sequences()

        iterations = max(1, self.config.get("iterations", 1))
        rcl_extra = max(0, self.config.get("rcl_extra", 0))
        rng = random.Random(self.config.get("random_seed", None))

        best_obj = -1.0
        best_total_units = -1
        best_orders: list[int] = []
        best_aisles: list[int] = []

        for _ in range(iterations):
            for k in range(1, max_k + 1):
                candidate_aisles = self._choose_aisles_for_k(
                    ranked_aisles, k, rcl_extra, rng
                )
                inventory_pool = self._pool_inventory(candidate_aisles)
                for order_sequence in order_sequences:
                    selected_orders, total_units = self._pack_orders_for_inventory(
                        inventory_pool, order_sequence
                    )

                    if total_units < self.lb:
                        continue

                    cleaned_aisles = self._cleanup_redundant_aisles(
                        selected_orders, candidate_aisles
                    )

                    if not cleaned_aisles:
                        continue

                    obj = total_units / len(cleaned_aisles)

                    is_better = obj > best_obj
                    same_obj_more_units = (
                        obj == best_obj and total_units > best_total_units
                    )
                    same_obj_same_units_less_aisles = (
                        obj == best_obj
                        and total_units == best_total_units
                        and (not best_aisles or len(cleaned_aisles) < len(best_aisles))
                    )

                    if (
                        is_better
                        or same_obj_more_units
                        or same_obj_same_units_less_aisles
                    ):
                        best_obj = obj
                        best_total_units = total_units
                        best_orders = selected_orders
                        best_aisles = cleaned_aisles

        if best_total_units < self.lb:
            return [], []

        return best_orders, best_aisles
