from models.solver import Solver

from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select


class GreedyOrderSimilarityAggregationHeuristic(Solver):

    def _build_total_stock(self) -> dict[int, int]:
        stock: dict[int, int] = {}

        for aisle in self.aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty

        return stock

    def _is_order_possible(self, order: dict[int, int], stock: dict[int, int]) -> bool:
        for item, qty in order.items():
            if stock.get(item, 0) < qty:
                return False
        return True

    def _order_units(self, order_idx: int) -> int:
        return sum(self.orders[order_idx].values())

    def _order_skus(self, order_idx: int) -> set[int]:
        return set(self.orders[order_idx].keys())

    def _pick_seed_order(self, stock: dict[int, int]) -> int | None:
        best_idx: int | None = None
        best_units = -1

        for idx in range(self.n_orders):
            units = self._order_units(idx)

            if units > self.ub:
                continue

            if not self._is_order_possible(self.orders[idx], stock):
                continue

            is_better = units > best_units
            same_units_lower_idx = units == best_units and (
                best_idx is None or idx < best_idx
            )

            if is_better or same_units_lower_idx:
                best_idx = idx
                best_units = units

        return best_idx

    def _build_demand(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}

        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        return demand

    def _is_demand_fully_covered(
        self, demand: dict[int, int], selected_aisles: list[int]
    ) -> bool:
        if not demand:
            return True

        inventory: dict[int, int] = {}

        for aisle_idx in selected_aisles:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty

        for item, qty in demand.items():
            if inventory.get(item, 0) < qty:
                return False

        return True

    def solve(self) -> tuple[list[int], list[int]]:
        total_stock = self._build_total_stock()

        seed_idx = self._pick_seed_order(total_stock)
        if seed_idx is None:
            return [], []

        selected_orders: list[int] = [seed_idx]
        selected_skus = self._order_skus(seed_idx)
        total_units = self._order_units(seed_idx)

        remaining_stock = dict(total_stock)

        for item, qty in self.orders[seed_idx].items():
            remaining_stock[item] = remaining_stock.get(item, 0) - qty

        unselected = set(range(self.n_orders))
        unselected.remove(seed_idx)

        while total_units < self.lb and unselected:
            best_idx: int | None = None
            best_similarity = -1
            best_units = -1

            for idx in unselected:
                order = self.orders[idx]
                order_units = self._order_units(idx)

                if total_units + order_units > self.ub:
                    continue

                if not self._is_order_possible(order, remaining_stock):
                    continue

                common_skus = len(selected_skus.intersection(order.keys()))

                is_better = common_skus > best_similarity
                tie_higher_units = (
                    common_skus == best_similarity and order_units > best_units
                )
                tie_lower_idx = (
                    common_skus == best_similarity
                    and order_units == best_units
                    and (best_idx is None or idx < best_idx)
                )

                if is_better or tie_higher_units or tie_lower_idx:
                    best_idx = idx
                    best_similarity = common_skus
                    best_units = order_units

            if best_idx is None:
                break

            selected_orders.append(best_idx)
            total_units += self._order_units(best_idx)
            selected_skus.update(self._order_skus(best_idx))

            for item, qty in self.orders[best_idx].items():
                remaining_stock[item] = remaining_stock.get(item, 0) - qty

            unselected.remove(best_idx)

        if total_units < self.lb:
            return [], []

        # Backtracking fallback: remove the last added order until aisle coverage is feasible.
        while selected_orders:
            demand = self._build_demand(selected_orders)
            selected_aisles = multi_greedy_aisle_select(demand, self.aisles)

            if self._is_demand_fully_covered(demand, selected_aisles):
                return selected_orders, selected_aisles

            removed_order_idx = selected_orders.pop()
            total_units -= self._order_units(removed_order_idx)

            if total_units < self.lb:
                return [], []

        return [], []
