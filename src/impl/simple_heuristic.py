from models.solver import Solver


class SimpleHeuristic(Solver):

    def solve(self) -> tuple[list[int], list[int]]:

        seed = self.config["seed"]

        if not seed:
            raise ValueError("Seed not provided in config for SimpleHeuristic")

        selected_orders: list[int] = []
        total_units = 0

        for order_idx in seed:
            order_units = sum(self.orders[order_idx].values())

            if total_units + order_units <= self.ub:
                selected_orders.append(order_idx)
                total_units += order_units

        if total_units < self.lb:
            return [], []

        demand: dict[int, int] = {}

        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        visited_aisles: set[int] = set()

        for item, needed in demand.items():
            supplied = sum(self.aisles[a].get(item, 0) for a in visited_aisles)

            if supplied >= needed:
                continue

            for aisle_idx in range(len(self.aisles)):
                if supplied >= needed:
                    break

                if (
                    aisle_idx not in visited_aisles
                    and self.aisles[aisle_idx].get(item, 0) > 0
                ):
                    visited_aisles.add(aisle_idx)
                    supplied += self.aisles[aisle_idx].get(item, 0)

        return selected_orders, sorted(visited_aisles)
