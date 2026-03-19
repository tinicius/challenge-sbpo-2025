from models.solver import Solver

from impl.utils.greedy_aisle_select import greedy_aisle_select
from impl.utils.similarity import similarity


class SimilarityHeuristic(Solver):

    def is_order_possible(self, order: dict[int, int], stock: dict[int, int]) -> bool:
        for idx, qnt in order.items():
            if stock.get(idx, 0) < qnt:
                return False
        return True

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self.config["seed"]
        reverse = self.config["reverse"]

        if not seed:
            raise ValueError("Seed not provided in config for SimilarHeuristic")

        stock: dict[int, int] = {}

        for aisle in self.aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty

        first_order = self.orders[seed[0]]

        sorted_orders = sorted(
            seed,
            key=lambda idx: similarity(first_order, self.orders[idx]),
            reverse=reverse,
        )

        total_units = 0
        selected_orders: list[int] = []

        for order_idx in sorted_orders:

            if not self.is_order_possible(self.orders[order_idx], stock):
                continue

            order_size = sum(self.orders[order_idx].values())

            size_restriction = total_units + order_size <= self.ub

            if size_restriction:
                selected_orders.append(order_idx)
                total_units += order_size

                for idx, qnt in self.orders[order_idx].items():
                    stock[idx] = stock.get(idx, 0) - qnt

        if total_units < self.lb:
            return [], []

        demand: dict[int, int] = {}

        for idx in selected_orders:
            for idx, qnt in self.orders[idx].items():
                demand[idx] = demand.get(idx, 0) + qnt

        visited_aisles = greedy_aisle_select(demand, self.aisles)

        return selected_orders, visited_aisles
