from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from models.solver import Solver


class MaxDensityOrderInsertionHeuristic(Solver):
    def _build_global_stock(self) -> dict[int, int]:
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

    def _compute_demand(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _estimate_new_aisles_required(
        self, order: dict[int, int], visited_aisles: set[int]
    ) -> int:
        if not order:
            return 0

        covered_by_visited: dict[int, int] = {}
        for aisle_idx in visited_aisles:
            for item, qty in self.aisles[aisle_idx].items():
                covered_by_visited[item] = covered_by_visited.get(item, 0) + qty

        residual_demand: dict[int, int] = {}
        for item, qty in order.items():
            missing = qty - covered_by_visited.get(item, 0)
            if missing > 0:
                residual_demand[item] = missing

        if not residual_demand:
            return 0

        unavailable_aisles = visited_aisles
        available_aisles = [
            aisle
            for aisle_idx, aisle in enumerate(self.aisles)
            if aisle_idx not in unavailable_aisles
        ]

        selected_local = multi_greedy_aisle_select(residual_demand, available_aisles)

        return len(selected_local)

    def _rank_remaining_orders(
        self,
        remaining: list[int],
        visited_aisles: set[int],
        order_units: list[int],
    ) -> list[int]:
        estimated_new_aisles = {
            order_idx: self._estimate_new_aisles_required(
                self.orders[order_idx], visited_aisles
            )
            for order_idx in remaining
        }

        ranked = sorted(
            remaining,
            key=lambda order_idx: (
                (
                    float("inf")
                    if estimated_new_aisles[order_idx] == 0
                    else (order_units[order_idx] / estimated_new_aisles[order_idx])
                ),
                order_units[order_idx],
                -order_idx,
            ),
            reverse=True,
        )
        return ranked

    def solve(self) -> tuple[list[int], list[int]]:
        stock = self._build_global_stock()

        order_units = [sum(order.values()) for order in self.orders]

        eligible_orders = [
            idx
            for idx in range(self.n_orders)
            if self._is_order_possible(self.orders[idx], stock)
        ]

        selected_orders: list[int] = []
        selected_set: set[int] = set()
        visited_aisles_estimate: set[int] = set()
        total_units = 0

        while total_units < self.lb and total_units < self.ub:
            remaining_orders = [
                idx for idx in eligible_orders if idx not in selected_set
            ]

            if not remaining_orders:
                break

            ranked_orders = self._rank_remaining_orders(
                remaining_orders,
                visited_aisles_estimate,
                order_units,
            )

            chosen_order = None
            for order_idx in ranked_orders:
                if total_units + order_units[order_idx] <= self.ub:
                    chosen_order = order_idx
                    break

            if chosen_order is None:
                break

            selected_orders.append(chosen_order)
            selected_set.add(chosen_order)
            total_units += order_units[chosen_order]

            demand = self._compute_demand(selected_orders)
            visited_aisles_estimate = set(
                multi_greedy_aisle_select(demand, self.aisles)
            )

            if total_units == self.ub:
                break

        if total_units < self.lb:
            return [], []

        demand = self._compute_demand(selected_orders)
        visited_aisles = multi_greedy_aisle_select(demand, self.aisles)

        return selected_orders, visited_aisles
