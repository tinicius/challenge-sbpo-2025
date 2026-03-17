from models.solver import Solver


class DiffHeuristic(Solver):

    def similarity(self, order1: dict[int, int], order2: dict[int, int]) -> float:
        # Calculate similarity based on Jaccard similarity of items in the orders

        set1 = set(order1.keys())
        set2 = set(order2.keys())

        intersection = set1.intersection(set2)
        union = set1.union(set2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self.config["seed"]

        if not seed:
            raise ValueError("Seed not provided in config for DiffHeuristic")

        first_order = self.orders[seed[0]]

        sorted_orders = sorted(
            seed,
            key=lambda idx: self.similarity(first_order, self.orders[idx]),
            reverse=False,
        )

        stock: dict = {}

        for aisle in self.aisles:

            for item_idx in aisle.keys():
                quantity = aisle[item_idx]

                stock[item_idx] = stock.get(item_idx, 0) + quantity

        total_units = 0
        selected_orders: list[int] = []

        for order_idx in sorted_orders:

            order_size = sum(self.orders[order_idx].values())

            size_restriction = total_units + order_size <= self.ub

            has_all_items = True

            for item, quantity in self.orders[order_idx].items():
                if stock[item] < quantity:
                    has_all_items = False
                    break

            if size_restriction and has_all_items:

                for item, quantity in self.orders[order_idx].items():
                    stock[item] -= quantity

                selected_orders.append(order_idx)
                total_units += order_size

        if total_units < self.lb:
            return [], []

        # Step 5: Greedy aisle selection (per-item demand accumulation)
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
