from models.solver import Solver


class SimilarHeuristic(Solver):

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
        # Idea: Select orders base read order (First Come First Serve) until wave size is within [lb, ub].

        seed = self.config["seed"]

        if not seed:
            raise ValueError("Seed not provided in config for SimilarHeuristic")

        first_order = self.orders[seed[0]]

        sorted_orders = sorted(
            seed,
            key=lambda idx: self.similarity(first_order, self.orders[idx]),
            reverse=True,
        )

        stock: dict = {}

        for aisle in self.aisles:

            for item_idx in aisle.keys():
                quantity = aisle[item_idx]

                stock[item_idx] = stock.get(item_idx, 0) + quantity

        count = 0
        selected_orders = []

        selected_aisles = set()

        for order_idx in sorted_orders:

            order_size = sum(self.orders[order_idx].values())

            size_restriction = count + order_size <= self.ub

            has_all_items = True

            for item, quantity in self.orders[order_idx].items():
                if stock[item] < quantity:
                    has_all_items = False
                    break

            if size_restriction and has_all_items:

                for item, quantity in self.orders[order_idx].items():
                    stock[item] -= quantity

                selected_orders.append(order_idx)
                count += order_size

        for order_idx in selected_orders:

            for item_order, quantity_order in self.orders[order_idx].items():

                for aisle_idx in range(len(self.aisles)):

                    aisle = self.aisles[aisle_idx]

                    for item_aisle in aisle.keys():

                        if item_order != item_aisle:
                            continue

                    if aisle[item_aisle] >= quantity_order:
                        aisle[item_order] = aisle[item_aisle] - quantity_order
                        selected_aisles.add(aisle_idx)

        return selected_orders, list(selected_aisles)
