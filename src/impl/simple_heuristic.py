from models.solver import Solver


class SimpleHeuristic(Solver):

    def _generate_seed(self) -> list[int]:
        """Return a seed list containing the index of the largest order by total units."""
        largest = max(range(self.n_orders), key=lambda i: sum(self.orders[i].values()))
        return [largest]

    def _similarity(self, items_a: set, items_b: set) -> float:
        """Jaccard similarity on item key sets. Swap this method to change the metric."""
        if not items_a and not items_b:
            return 0.0
        return len(items_a & items_b) / len(items_a | items_b)

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self._generate_seed()
        seed_set = set(seed)

        # --- Phase 1: initialise wave from seed orders ---
        selected_orders: list[int] = []
        total_units = 0
        ref_items: set = set()

        for order_idx in seed:
            order = self.orders[order_idx]
            order_units = sum(order.values())
            if total_units + order_units <= self.ub:
                selected_orders.append(order_idx)
                total_units += order_units
                ref_items.update(order.keys())

        # --- Phase 2: rank remaining orders by similarity to ref_items ---
        similarities: list[tuple[float, int]] = []
        for order_idx in range(self.n_orders):
            if order_idx in seed_set:
                continue
            order_items = set(self.orders[order_idx].keys())
            sim = self._similarity(ref_items, order_items)
            similarities.append((sim, order_idx))

        similarities.sort(key=lambda x: x[0], reverse=True)

        # --- Phase 3: greedy fill up to UB ---
        for _sim, order_idx in similarities:
            order = self.orders[order_idx]
            order_units = sum(order.values())
            if total_units + order_units <= self.ub:
                selected_orders.append(order_idx)
                total_units += order_units

        # --- Phase 4: LB feasibility check ---
        if total_units < self.lb:
            return ([], [])

        # --- Phase 5: greedy aisle selection ---
        total_demand: dict = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                total_demand[item] = total_demand.get(item, 0) + qty

        visited_aisles: set = set()
        current_supply: dict = {}

        for item_id, needed in total_demand.items():
            if current_supply.get(item_id, 0) >= needed:
                continue
            for aisle_idx in range(self.n_aisles):
                if aisle_idx in visited_aisles:
                    continue
                if self.aisles[aisle_idx].get(item_id, 0) > 0:
                    visited_aisles.add(aisle_idx)
                    for itm, stk in self.aisles[aisle_idx].items():
                        current_supply[itm] = current_supply.get(itm, 0) + stk
                    if current_supply.get(item_id, 0) >= needed:
                        break

        return (selected_orders, sorted(visited_aisles))

