from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.shuffled_indexes import shuffled_indexes
from problems.base import ProblemInput
from algorithms.utils.similarity import similarity


class OrderAisleSimilarity(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "order_aisle_similarity"

    def check_possible(self, orders: list[int], aisles: list[int]) -> bool:

        stock: dict[int, int] = {}

        for aisle_idx in aisles:
            for item, qty in self.aisles[aisle_idx].items():
                stock[item] = stock.get(item, 0) + qty

        demand: dict[int, int] = {}
        for order_idx in orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        for item, qty in demand.items():
            if stock.get(item, 0) < qty:
                return False

        return True

    def is_order_possible(self, order: dict[int, int], stock: dict[int, int]) -> bool:
        for idx, qnt in order.items():
            if stock.get(idx, 0) < qnt:
                return False
        return True

    def prune_aisles(self, orders: list[int], aisles: list[int]) -> list[int]:
        pruned = aisles.copy()
        for aisle_idx in reversed(aisles):
            candidate = [a for a in pruned if a != aisle_idx]
            if candidate and self.check_possible(orders, candidate):
                pruned = candidate
        return pruned

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        shuffled_orders = shuffled_indexes(self.n_orders)

        stock: dict[int, int] = {}

        for aisle in self.aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty

        best_obj = -1
        best_selected_orders = []
        best_visited_aisles = []

        for k in range(0, 10):
            selected_orders: list[int] = []
            selected_aisles: list[int] = []

            seed_order = self.orders[shuffled_orders[k]]

            sorted_orders_idx = sorted(
                range(self.n_orders),
                key=lambda idx: similarity(self.orders[idx], seed_order),
                reverse=True,
            )

            sorted_aisles_idx = sorted(
                range(self.n_aisles),
                key=lambda idx: similarity(self.aisles[idx], seed_order),
                reverse=True,
            )

            actual_order = 0
            actual_aisle = 0

            total_items = 0
            agg_demand: dict[int, int] = {}

            for order_idx in sorted_orders_idx:

                if not self.is_order_possible(self.orders[order_idx], stock):
                    continue

                order_items = sum(self.orders[order_idx].values())

                if total_items + order_items > self.ub:
                    continue

                can_add = True
                for item, qty in self.orders[order_idx].items():
                    if agg_demand.get(item, 0) + qty > stock.get(item, 0):
                        can_add = False
                        break
                if not can_add:
                    continue

                selected_orders.append(order_idx)
                total_items += order_items
                for item, qty in self.orders[order_idx].items():
                    agg_demand[item] = agg_demand.get(item, 0) + qty
                actual_order += 1

                if total_items < self.lb:
                    continue

                is_possible = False

                while not is_possible and actual_aisle < self.n_aisles:
                    selected_aisles.append(sorted_aisles_idx[actual_aisle])
                    actual_aisle += 1

                    is_possible = self.check_possible(selected_orders, selected_aisles)

                if is_possible:
                    pruned_aisles = self.prune_aisles(selected_orders, selected_aisles)
                    obj = total_items / len(pruned_aisles)

                    if obj > best_obj:
                        best_obj = obj
                        best_selected_orders = selected_orders.copy()
                        best_visited_aisles = pruned_aisles

        return {
            "selected_orders": best_selected_orders,
            "visited_aisles": best_visited_aisles,
            "objective": best_obj,
        }
