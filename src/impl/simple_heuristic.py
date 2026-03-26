from models.solver import Solver
from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from impl.utils.greedy_aisle_select import greedy_aisle_select
from impl.utils.table import write_dict_table_to_file
from impl.utils.shuffled_indexes import shuffled_indexes


class SimpleHeuristic(Solver):

    def is_order_possible(self, order: dict[int, int], stock: dict[int, int]) -> bool:
        for idx, qnt in order.items():
            if stock.get(idx, 0) < qnt:
                return False
        return True

    def solve(self) -> tuple[list[int], list[int]]:

        seed = shuffled_indexes(self.n_orders)

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
            return [], []

        demand: dict[int, int] = {}

        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        visited_aisles = []

        if greedy == "multi":
            visited_aisles = multi_greedy_aisle_select(demand, self.aisles)
        else:
            visited_aisles = greedy_aisle_select(demand, self.aisles)

        # selected_stock: dict[int, int] = {}

        # for idx in visited_aisles:
        #     aisle = self.aisles[idx]
        #     for item, qty in aisle.items():
        #         selected_stock[item] = selected_stock.get(item, 0) + qty

        # write_dict_table_to_file(
        #     selected_stock, demand, missing_val="-", filename="comparison_table.txt"
        # )
        # write_dict_table_to_file(
        #     stock, demand, missing_val="-", filename="comparison_table.txt"
        # )

        return selected_orders, visited_aisles
