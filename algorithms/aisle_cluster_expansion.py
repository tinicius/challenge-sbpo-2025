import random

from algorithms.base import Algorithm
from problems.base import ProblemInput


class AisleClusterExpansion(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "aisle_cluster_expansion"

    def _order_units(self) -> list[int]:
        return [sum(order.values()) for order in self.orders]

    def _inventory_for_aisles(self, aisle_indexes: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indexes:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _order_covered_units(self, order_idx: int, inventory: dict[int, int]) -> int:
        covered = 0
        for item, qty in self.orders[order_idx].items():
            covered += min(qty, inventory.get(item, 0))
        return covered

    def _is_order_fully_coverable(
        self, order_idx: int, inventory: dict[int, int]
    ) -> bool:
        for item, qty in self.orders[order_idx].items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _pack_orders(
        self,
        candidate_orders: list[int],
        inventory: dict[int, int],
        order_units: list[int],
        prefer_large: bool,
    ) -> tuple[list[int], int]:
        remaining_inventory = dict(inventory)
        selected_orders: list[int] = []
        total_units = 0

        sorted_orders = sorted(
            candidate_orders,
            key=lambda idx: (
                -order_units[idx] if prefer_large else order_units[idx],
                idx,
            ),
        )

        for order_idx in sorted_orders:
            order = self.orders[order_idx]
            units = order_units[order_idx]

            if total_units + units > self.ub:
                continue

            feasible = True
            for item, qty in order.items():
                if remaining_inventory.get(item, 0) < qty:
                    feasible = False
                    break

            if not feasible:
                continue

            selected_orders.append(order_idx)
            total_units += units

            for item, qty in order.items():
                remaining_inventory[item] = remaining_inventory.get(item, 0) - qty

        return selected_orders, total_units

    def _best_feasible_subset(
        self,
        candidate_orders: list[int],
        inventory: dict[int, int],
        order_units: list[int],
    ) -> tuple[list[int], int]:
        best_orders: list[int] = []
        best_units = -1

        for prefer_large in [True, False]:
            packed_orders, packed_units = self._pack_orders(
                candidate_orders, inventory, order_units, prefer_large
            )

            if packed_units < self.lb:
                continue

            if packed_units > best_units:
                best_orders = packed_orders
                best_units = packed_units

        if best_units < self.lb:
            return [], -1

        return best_orders, best_units

    def _missing_demand_from_partial_orders(
        self,
        partial_orders: list[int],
        inventory: dict[int, int],
    ) -> dict[int, int]:
        missing: dict[int, int] = {}

        for order_idx in partial_orders:
            for item, qty in self.orders[order_idx].items():
                deficit = qty - inventory.get(item, 0)
                if deficit > 0:
                    missing[item] = missing.get(item, 0) + deficit

        return missing

    def _score_aisle_for_missing(
        self,
        aisle_idx: int,
        missing_demand: dict[int, int],
    ) -> int:
        score = 0
        for item, qty in self.aisles[aisle_idx].items():
            score += min(qty, missing_demand.get(item, 0))
        return score

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        if self.n_aisles == 0 or self.n_orders == 0:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        order_units = self._order_units()

        attempts_cfg = self.config.get("attempts", self.n_aisles)
        attempts = max(1, min(attempts_cfg, self.n_aisles))

        max_added_aisles_cfg = self.config.get("max_added_aisles", self.n_aisles - 1)
        max_added_aisles = max(0, min(max_added_aisles_cfg, self.n_aisles - 1))

        rng = random.SystemRandom()
        seed_candidates = rng.sample(list(range(self.n_aisles)), self.n_aisles)

        best_obj = -1.0
        best_units = -1
        best_orders: list[int] = []
        best_aisles: list[int] = []

        for seed_aisle in seed_candidates[:attempts]:
            current_aisles = [seed_aisle]
            added_aisles = 0

            while True:
                inventory = self._inventory_for_aisles(current_aisles)

                touched_orders: list[int] = []
                full_orders: list[int] = []
                partial_orders: list[int] = []

                for order_idx in range(self.n_orders):
                    covered_units = self._order_covered_units(order_idx, inventory)

                    if covered_units == 0:
                        continue

                    touched_orders.append(order_idx)

                    if covered_units == order_units[
                        order_idx
                    ] and self._is_order_fully_coverable(order_idx, inventory):
                        full_orders.append(order_idx)
                    else:
                        partial_orders.append(order_idx)

                if touched_orders and full_orders:
                    selected_orders, total_units = self._best_feasible_subset(
                        full_orders,
                        inventory,
                        order_units,
                    )

                    if selected_orders:
                        objective = total_units / len(current_aisles)

                        better_obj = objective > best_obj
                        same_obj_more_units = (
                            objective == best_obj and total_units > best_units
                        )
                        same_obj_same_units_less_aisles = (
                            objective == best_obj
                            and total_units == best_units
                            and (
                                not best_aisles
                                or len(current_aisles) < len(best_aisles)
                            )
                        )

                        if (
                            better_obj
                            or same_obj_more_units
                            or same_obj_same_units_less_aisles
                        ):
                            best_obj = objective
                            best_units = total_units
                            best_orders = selected_orders
                            best_aisles = sorted(current_aisles)

                        # First feasible subset reached for this seed.
                        break

                if added_aisles >= max_added_aisles:
                    break

                missing_demand = self._missing_demand_from_partial_orders(
                    partial_orders,
                    inventory,
                )

                if not missing_demand:
                    break

                available_aisles = [
                    aisle_idx
                    for aisle_idx in range(self.n_aisles)
                    if aisle_idx not in current_aisles
                ]

                best_next_aisle = -1
                best_next_score = 0

                for aisle_idx in available_aisles:
                    score = self._score_aisle_for_missing(aisle_idx, missing_demand)
                    if score > best_next_score:
                        best_next_score = score
                        best_next_aisle = aisle_idx

                if best_next_score <= 0:
                    break

                current_aisles.append(best_next_aisle)
                added_aisles += 1

        if best_units < self.lb or not best_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        if not best_orders or not best_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in best_orders)
        objective = total_items / len(best_aisles)
        return {'selected_orders': best_orders, 'visited_aisles': best_aisles, 'objective': objective}
