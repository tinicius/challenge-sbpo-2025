from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.similarity import similarity
from algorithms.utils.shuffled_indexes import shuffled_indexes
from problems.base import ProblemInput


class SimilarityHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "similarity_heuristic"

    def is_order_possible(self, order: dict[int, int], stock: dict[int, int]) -> bool:
        for idx, qnt in order.items():
            if stock.get(idx, 0) < qnt:
                return False
        return True

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        reverse = self.config["reverse"]

        greedy = self.config["greedy"]

        if not greedy:
            raise ValueError(
                "Greedy flag not provided in config for SimilarityHeuristic"
            )

        seed = shuffled_indexes(self.n_orders)

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
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        demand: dict[int, int] = {}

        for idx in selected_orders:
            for idx, qnt in self.orders[idx].items():
                demand[idx] = demand.get(idx, 0) + qnt

        visited_aisles = []

        if greedy == "multi":
            visited_aisles = multi_greedy_aisle_select(demand, self.aisles)
        else:
            visited_aisles = greedy_aisle_select(demand, self.aisles)

        if not selected_orders or not visited_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}
