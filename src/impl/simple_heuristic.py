from models.solver import Solver


class SimpleHeuristic(Solver):

    def _similarity(self, items_a: set, items_b: set) -> float:
        """Jaccard similarity between two item key sets."""
        if not items_a and not items_b:
            return 0.0
        union = len(items_a | items_b)
        return len(items_a & items_b) / union if union > 0 else 0.0

    def solve(self) -> tuple[list[int], list[int]]:
        # Implements the Order Similarity Seed Algorithm described in docs/simple_heuristic.md

        seed = self.config["seed"]

        if not seed:
            raise ValueError("Seed not provided in config for SimpleHeuristic")

        # Step 1: Init wave from seed
        selected_orders: list[int] = []
        total_units = 0
        ref_items: set[int] = set()

        for order_idx in seed:
            order_units = sum(self.orders[order_idx].values())
            if total_units + order_units <= self.ub:
                selected_orders.append(order_idx)
                total_units += order_units
                ref_items.update(self.orders[order_idx].keys())

        # Step 2: Rank remaining orders by Jaccard similarity to ref_items
        seed_set = set(seed)
        remaining = [
            (self._similarity(ref_items, set(self.orders[o].keys())), o)
            for o in range(self.n_orders)
            if o not in seed_set
        ]
        remaining.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Greedy fill from remaining orders (highest similarity first)
        for _sim, order_idx in remaining:
            order_units = sum(self.orders[order_idx].values())
            if total_units + order_units <= self.ub:
                selected_orders.append(order_idx)
                total_units += order_units

        # Step 4: Feasibility check — wave must meet lower bound
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
                if aisle_idx not in visited_aisles and self.aisles[aisle_idx].get(item, 0) > 0:
                    visited_aisles.add(aisle_idx)
                    supplied += self.aisles[aisle_idx].get(item, 0)

        return selected_orders, sorted(visited_aisles)
