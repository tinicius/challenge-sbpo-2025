from models.solver import Solver


class GreedyAisleDensityHeuristic(Solver):
    def _rank_aisles_by_density(self) -> list[int]:
        ranked = []
        for aisle_idx, aisle in enumerate(self.aisles):
            density = sum(aisle.values())
            ranked.append((aisle_idx, density))

        ranked.sort(key=lambda x: (-x[1], x[0]))
        return [aisle_idx for aisle_idx, _ in ranked]

    def _pool_inventory(self, aisle_indices: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _allocate_orders_with_consumption(
        self, aisle_indices: list[int]
    ) -> tuple[list[int], int]:
        inventory = self._pool_inventory(aisle_indices)
        selected_orders: list[int] = []
        total_units = 0

        for order_idx, order in enumerate(self.orders):
            feasible = True
            for item, qty in order.items():
                if inventory.get(item, 0) < qty:
                    feasible = False
                    break

            if not feasible:
                continue

            selected_orders.append(order_idx)
            order_units = sum(order.values())
            total_units += order_units

            for item, qty in order.items():
                inventory[item] = inventory.get(item, 0) - qty

        return selected_orders, total_units

    def _order_units(self, order_idx: int) -> int:
        return sum(self.orders[order_idx].values())

    def _enforce_upper_bound(
        self, selected_orders: list[int], total_units: int
    ) -> tuple[list[int], int]:
        if total_units <= self.ub:
            return list(selected_orders), total_units

        reduced_orders = list(selected_orders)

        # Remove smaller orders first; ties remove the highest index order.
        removal_order = sorted(
            reduced_orders,
            key=lambda order_idx: (self._order_units(order_idx), -order_idx),
        )

        for order_idx in removal_order:
            if total_units <= self.ub:
                break

            reduced_orders.remove(order_idx)
            total_units -= self._order_units(order_idx)

        return reduced_orders, total_units

    def _build_demand_from_orders(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _demand_is_coverable_by_aisles(
        self, demand: dict[int, int], aisle_indices: list[int]
    ) -> bool:
        inventory = self._pool_inventory(aisle_indices)
        for item, qty in demand.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _cleanup_redundant_aisles(
        self, selected_orders: list[int], aisle_indices: list[int]
    ) -> list[int]:
        if not selected_orders:
            return []

        demand = self._build_demand_from_orders(selected_orders)
        cleaned = sorted(set(aisle_indices))

        for aisle_idx in list(cleaned):
            if len(cleaned) == 1:
                break

            candidate = [idx for idx in cleaned if idx != aisle_idx]
            if self._demand_is_coverable_by_aisles(demand, candidate):
                cleaned = candidate

        return cleaned

    def solve(self) -> tuple[list[int], list[int]]:
        ranked_aisles = self._rank_aisles_by_density()
        if not ranked_aisles:
            return [], []

        initial_k = max(1, int(self.config.get("initial_k", 1)))
        max_k = min(
            self.n_aisles, max(initial_k, int(self.config.get("max_k", self.n_aisles)))
        )

        for k in range(initial_k, max_k + 1):
            candidate_aisles = ranked_aisles[:k]
            selected_orders, total_units = self._allocate_orders_with_consumption(
                candidate_aisles
            )

            selected_orders, total_units = self._enforce_upper_bound(
                selected_orders, total_units
            )

            if total_units < self.lb:
                continue

            cleaned_aisles = self._cleanup_redundant_aisles(
                selected_orders, candidate_aisles
            )

            if not cleaned_aisles:
                continue

            return selected_orders, cleaned_aisles

        return [], []
