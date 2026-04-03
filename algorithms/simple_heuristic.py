from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.shuffled_indexes import shuffled_indexes
from problems.base import ProblemInput


class SimpleHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "simple_heuristic"

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

        seed = shuffled_indexes(self.n_orders)

        if self.config.get("order") is not None:
            order = self.config["order"]

            seed = sorted(
                seed,
                key=lambda idx: sum(self.orders[idx].values()),
                reverse=(order == "asc"),
            )

        greedy = self.config["greedy"]

        if not greedy:
            raise ValueError(
                "Greedy flag not provided in config for SimilarityHeuristic"
            )

        stock: dict[int, int] = {}

        for aisle in self.aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty

        selected_orders: list[int] = []
        total_units = 0

        for order_idx in seed:

            if not self.is_order_possible(self.orders[order_idx], stock):
                continue

            order_units = sum(self.orders[order_idx].values())

            if total_units + order_units <= self.ub:
                selected_orders.append(order_idx)
                total_units += order_units

                for idx, qnt in self.orders[order_idx].items():
                    stock[idx] = stock.get(idx, 0) - qnt

        if total_units < self.lb:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        demand: dict[int, int] = {}

        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        visited_aisles = []

        if greedy == "multi":
            visited_aisles = multi_greedy_aisle_select(demand, self.aisles)
        else:
            visited_aisles = greedy_aisle_select(demand, self.aisles)

        if not selected_orders or not visited_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": objective,
        }
