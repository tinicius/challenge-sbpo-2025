from models.solver import Solver


class Greedy(Solver):

    def solve(self) -> tuple[list[int], list[int]]:
        """
            Simple greedy solver for the Wave Order Picking problem.

            Iterates through orders sequentially. For each order added, accumulates
            item demand and greedily selects aisles to cover every item's demand.
            Stops when total units picked falls within [lb, ub].
        """

        selected_orders = []
        total_demand = {}  # item_id -> total quantity needed
        total_units = 0

        for order_idx in range(self.n_orders):
            order = self.orders[order_idx]
            order_units = sum(order.values())

            # If adding this order would exceed ub, skip it
            if total_units + order_units > self.ub:
                continue

            # Tentatively add this order
            selected_orders.append(order_idx)
            total_units += order_units

            for item_id, qty in order.items():
                total_demand[item_id] = total_demand.get(item_id, 0) + qty

            # Check if we're within bounds
            if total_units >= self.lb:
                break

        # Now greedily select aisles to cover all item demands
        visited_aisles = set()
        # item_id -> remaining qty needed
        remaining_demand = dict(total_demand)

        for item_id, needed in sorted(remaining_demand.items()):
            supplied = 0
            # Sum up what's already supplied by already-visited aisles
            for aisle_idx in visited_aisles:
                supplied += self.aisles[aisle_idx].get(item_id, 0)

            if supplied >= needed:
                continue

            # Need more supply for this item — iterate aisles to find ones that have it
            for aisle_idx in range(self.n_aisles):
                if aisle_idx in visited_aisles:
                    continue
                aisle_stock = self.aisles[aisle_idx].get(item_id, 0)
                if aisle_stock > 0:
                    visited_aisles.add(aisle_idx)
                    supplied += aisle_stock
                    if supplied >= needed:
                        break

        return list(selected_orders), sorted(visited_aisles)
