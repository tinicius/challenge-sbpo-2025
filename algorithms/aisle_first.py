from algorithms.base import Algorithm
from problems.base import ProblemInput


class AisleFirstHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "aisle_first"

    def _build_total_demand(self) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order in self.orders:
            for item, qty in order.items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _score_aisles(self, total_demand: dict[int, int]) -> list[tuple[int, int]]:
        scored: list[tuple[int, int]] = []
        for aisle_idx, aisle in enumerate(self.aisles):
            useful_inventory = 0
            for item, qty in aisle.items():
                useful_inventory += min(qty, total_demand.get(item, 0))
            scored.append((aisle_idx, useful_inventory))

        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored

    def _pool_inventory(self, aisle_indices: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _is_order_feasible_against_inventory(
        self, order: dict[int, int], inventory: dict[int, int]
    ) -> bool:
        for item, qty in order.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _build_order_sequences(self) -> list[int]:
        strategy = self.config.get("order_strategy", "size_asc")

        base_list = list(range(self.n_orders))

        if strategy == "random":
            return base_list

        asc = sorted(base_list, key=lambda idx: (sum(self.orders[idx].values()), idx))
        desc = sorted(base_list, key=lambda idx: (-sum(self.orders[idx].values()), idx))

        if strategy == "size_asc":
            return asc

        if strategy == "size_desc":
            return desc

        raise ValueError(f"Invalid order strategy: {strategy}")

    def _build_demand_from_orders(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _demand_is_coverable_by_aisles(
        self, demand: dict[int, int], aisle_indices: list[int]
    ) -> bool:
        if not demand:
            return True

        inventory = self._pool_inventory(aisle_indices)
        for item, qty in demand.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _cleanup_redundant_aisles(
        self, selected_orders: list[int], aisle_indices: list[int]
    ) -> list[int]:
        if not selected_orders or not aisle_indices:
            return []

        demand = self._build_demand_from_orders(selected_orders)
        current_aisles = sorted(set(aisle_indices))

        # Remove low-contribution aisles first to shrink denominator faster.
        scored_aisles: list[tuple[int, int]] = []
        for aisle_idx in current_aisles:
            contribution = sum(
                min(qty, demand.get(item, 0))
                for item, qty in self.aisles[aisle_idx].items()
            )
            scored_aisles.append((aisle_idx, contribution))

        for aisle_idx, _ in sorted(scored_aisles, key=lambda x: (x[1], x[0])):
            if len(current_aisles) == 1:
                break

            candidate_aisles = [idx for idx in current_aisles if idx != aisle_idx]
            if self._demand_is_coverable_by_aisles(demand, candidate_aisles):
                current_aisles = candidate_aisles

        return current_aisles

    def _choose_aisles_for_k(
        self,
        seed_aisle: int,
        similar_aisles: list[int],
        k: int,
    ) -> list[int]:
        if k <= 0:
            return []

        selected = [seed_aisle]
        for aisle_idx in similar_aisles:
            if len(selected) >= k:
                break
            selected.append(aisle_idx)

        selected.sort()
        return selected

    def _pack_orders_for_inventory(
        self, inventory_pool: dict[int, int], order_sequence: list[int]
    ) -> tuple[list[int], int]:
        selected_orders: list[int] = []
        total_units = 0

        remaining_inventory = dict(inventory_pool)

        for order_idx in order_sequence:
            order = self.orders[order_idx]
            order_units = sum(order.values())

            if total_units + order_units > self.ub:
                continue

            if not self._is_order_feasible_against_inventory(
                order, remaining_inventory
            ):
                continue

            selected_orders.append(order_idx)
            total_units += order_units

            for item, qty in order.items():
                remaining_inventory[item] = remaining_inventory.get(item, 0) - qty

        return selected_orders, total_units

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        total_demand = self._build_total_demand()
        ranked_scores = self._score_aisles(total_demand)
        ranked_aisles = [aisle_idx for aisle_idx, _ in ranked_scores]

        if not ranked_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        max_seed_aisles_cfg = self.config.get("max_k", 1)

        max_seed_aisles = max(1, min(max_seed_aisles_cfg, self.n_aisles))

        order_sequence = self._build_order_sequences()

        seed_aisles = ranked_aisles[:max_seed_aisles]

        rank_pos = {aisle_idx: pos for pos, aisle_idx in enumerate(ranked_aisles)}

        aisle_items = [set(self.aisles[idx].keys()) for idx in range(self.n_aisles)]

        sorted_similar_aisles_by_seed: dict[int, list[int]] = {}

        for seed_aisle in seed_aisles:
            seed_items = aisle_items[seed_aisle]
            seed_similarities: dict[int, float] = {}
            for other_aisle in ranked_aisles:
                if other_aisle == seed_aisle:
                    continue
                other_items = aisle_items[other_aisle]
                seed_similarities[other_aisle] = self._compute_jaccard_similarity(
                    seed_items, other_items
                )
            sorted_similar_aisles_by_seed[seed_aisle] = sorted(
                seed_similarities.keys(),
                # Tie-break by global aisle ranking (higher useful inventory first).
                key=lambda idx: (seed_similarities[idx], -rank_pos[idx]),
                reverse=True,
            )

        best_obj = -1.0
        best_total_units = -1
        best_orders: list[int] = []
        best_aisles: list[int] = []

        for seed_aisle in seed_aisles:
            max_aisles_for_seed = min(
                max_seed_aisles, len(sorted_similar_aisles_by_seed[seed_aisle]) + 1
            )
            for k in range(max_aisles_for_seed, 0, -1):
                candidate_aisles = self._choose_aisles_for_k(
                    seed_aisle, sorted_similar_aisles_by_seed[seed_aisle], k
                )
                inventory_pool = self._pool_inventory(candidate_aisles)

                selected_orders, total_units = self._pack_orders_for_inventory(
                    inventory_pool, order_sequence
                )

                if total_units < self.lb:
                    continue

                cleaned_aisles = self._cleanup_redundant_aisles(
                    selected_orders, candidate_aisles
                )

                if not cleaned_aisles:
                    continue

                obj = total_units / len(cleaned_aisles)

                is_better = obj > best_obj
                same_obj_more_units = obj == best_obj and total_units > best_total_units
                same_obj_same_units_less_aisles = (
                    obj == best_obj
                    and total_units == best_total_units
                    and (not best_aisles or len(cleaned_aisles) < len(best_aisles))
                )

                if is_better or same_obj_more_units or same_obj_same_units_less_aisles:
                    best_obj = obj
                    best_total_units = total_units
                    best_orders = selected_orders
                    best_aisles = cleaned_aisles

        selected_orders = best_orders
        visited_aisles = best_aisles

        if best_total_units < self.lb:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        if not selected_orders or not visited_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": objective,
        }

    def _compute_jaccard_similarity(
        self, first_items: set[int], second_items: set[int]
    ) -> float:
        intersection_size = len(first_items.intersection(second_items))
        union_size = len(first_items) + len(second_items) - intersection_size
        if union_size == 0:
            return 0.0
        return intersection_size / union_size
