import random

from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from models.solver import Solver


class PingPongAlternatingHeuristic(Solver):
    def _order_units(self, order_idx: int) -> int:
        return sum(self.orders[order_idx].values())

    def _total_units(self, selected_orders: list[int]) -> int:
        return sum(self._order_units(order_idx) for order_idx in selected_orders)

    def _build_demand(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _pool_inventory(self, aisle_indices: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _is_demand_coverable_by_inventory(
        self, demand: dict[int, int], inventory: dict[int, int]
    ) -> bool:
        for item, qty in demand.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _is_order_coverable_with_demand(
        self,
        order_idx: int,
        demand: dict[int, int],
        inventory: dict[int, int],
    ) -> bool:
        order = self.orders[order_idx]
        for item, qty in order.items():
            if demand.get(item, 0) + qty > inventory.get(item, 0):
                return False
        return True

    def _demand_is_coverable_by_aisles(
        self, demand: dict[int, int], aisle_indices: list[int]
    ) -> bool:
        inventory = self._pool_inventory(aisle_indices)
        return self._is_demand_coverable_by_inventory(demand, inventory)

    def _cleanup_redundant_aisles(
        self, selected_orders: list[int], aisle_indices: list[int]
    ) -> list[int]:
        if not selected_orders or not aisle_indices:
            return []

        demand = self._build_demand(selected_orders)
        cleaned = sorted(set(aisle_indices))

        scored_aisles: list[tuple[int, int]] = []
        for aisle_idx in cleaned:
            contribution = sum(
                min(qty, demand.get(item, 0))
                for item, qty in self.aisles[aisle_idx].items()
            )
            scored_aisles.append((aisle_idx, contribution))

        for aisle_idx, _ in sorted(scored_aisles, key=lambda x: (x[1], x[0])):
            if len(cleaned) == 1:
                break

            candidate = [idx for idx in cleaned if idx != aisle_idx]
            if self._demand_is_coverable_by_aisles(demand, candidate):
                cleaned = candidate

        return cleaned

    def _construct_greedy_fallback(self) -> tuple[list[int], list[int]]:
        global_stock: dict[int, int] = {}
        for aisle in self.aisles:
            for item, qty in aisle.items():
                global_stock[item] = global_stock.get(item, 0) + qty

        selected_orders: list[int] = []
        total_units = 0

        ranked_orders = sorted(
            range(self.n_orders),
            key=lambda idx: (-self._order_units(idx), idx),
        )

        for order_idx in ranked_orders:
            order = self.orders[order_idx]
            order_units = self._order_units(order_idx)

            if total_units + order_units > self.ub:
                continue

            feasible = True
            for item, qty in order.items():
                if global_stock.get(item, 0) < qty:
                    feasible = False
                    break

            if not feasible:
                continue

            selected_orders.append(order_idx)
            total_units += order_units

            for item, qty in order.items():
                global_stock[item] = global_stock.get(item, 0) - qty

        if total_units < self.lb:
            return [], []

        demand = self._build_demand(selected_orders)
        selected_aisles = multi_greedy_aisle_select(dict(demand), self.aisles)
        if not selected_aisles:
            return [], []

        cleaned_aisles = self._cleanup_redundant_aisles(
            selected_orders, selected_aisles
        )
        if not cleaned_aisles:
            return [], []

        return selected_orders, cleaned_aisles

    def _build_phase1_order_sequence(
        self,
        mode: str,
        rng: random.Random,
        base_orders: list[int],
    ) -> list[int]:
        if mode == "random":
            remaining = [
                idx for idx in range(self.n_orders) if idx not in set(base_orders)
            ]
            rng.shuffle(remaining)
            return list(base_orders) + remaining

        if mode == "greedy":
            ranked = sorted(
                range(self.n_orders),
                key=lambda idx: (-self._order_units(idx), idx),
            )
            base_set = set(base_orders)
            tail = [idx for idx in ranked if idx not in base_set]
            return list(base_orders) + tail

        ranked = sorted(
            range(self.n_orders),
            key=lambda idx: (-self._order_units(idx), idx),
        )
        base_set = set(base_orders)
        tail = [idx for idx in ranked if idx not in base_set]
        return list(base_orders) + tail

    def _pack_orders_to_bounds(
        self, order_sequence: list[int]
    ) -> tuple[list[int], int]:
        selected: list[int] = []
        total_units = 0

        for order_idx in order_sequence:
            units = self._order_units(order_idx)

            if total_units + units > self.ub:
                continue

            selected.append(order_idx)
            total_units += units

        return selected, total_units

    def _phase1_order_to_aisle(
        self,
        mode: str,
        rng: random.Random,
        base_orders: list[int],
    ) -> tuple[list[int], list[int], int]:
        sequence = self._build_phase1_order_sequence(mode, rng, base_orders)
        selected_orders, total_units = self._pack_orders_to_bounds(sequence)

        if total_units < self.lb:
            return [], [], -1

        demand = self._build_demand(selected_orders)
        if not demand:
            return [], [], -1

        selected_aisles = multi_greedy_aisle_select(dict(demand), self.aisles)
        if not selected_aisles:
            return [], [], -1

        if not self._demand_is_coverable_by_aisles(demand, selected_aisles):
            return [], [], -1

        cleaned_aisles = self._cleanup_redundant_aisles(
            selected_orders, selected_aisles
        )
        if not cleaned_aisles:
            return [], [], -1

        return selected_orders, cleaned_aisles, total_units

    def _phase2_aisle_to_order(
        self,
        selected_orders: list[int],
        selected_aisles: list[int],
    ) -> tuple[list[int], int]:
        if not selected_orders or not selected_aisles:
            return selected_orders, self._total_units(selected_orders)

        inventory = self._pool_inventory(selected_aisles)
        demand = self._build_demand(selected_orders)
        total_units = self._total_units(selected_orders)

        order_sequence = sorted(
            [idx for idx in range(self.n_orders) if idx not in set(selected_orders)],
            key=lambda idx: (-self._order_units(idx), idx),
        )

        expanded_orders = list(selected_orders)

        for order_idx in order_sequence:
            order_units = self._order_units(order_idx)
            if total_units + order_units > self.ub:
                continue

            if not self._is_order_coverable_with_demand(order_idx, demand, inventory):
                continue

            expanded_orders.append(order_idx)
            total_units += order_units

            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        return expanded_orders, total_units

    def _phase3_pruning(
        self,
        selected_orders: list[int],
        selected_aisles: list[int],
        current_units: int,
        eps: float,
    ) -> tuple[list[int], list[int], int]:
        if not selected_orders or not selected_aisles:
            return [], [], -1

        cur_orders = list(selected_orders)
        cur_aisles = self._cleanup_redundant_aisles(cur_orders, selected_aisles)
        if not cur_aisles:
            return [], [], -1

        cur_units = current_units
        cur_obj = cur_units / len(cur_aisles)

        while True:
            improved = False

            candidate_order_indexes = sorted(
                cur_orders,
                key=lambda idx: (self._order_units(idx), idx),
            )

            for order_idx in candidate_order_indexes:
                candidate_orders = [idx for idx in cur_orders if idx != order_idx]
                candidate_units = cur_units - self._order_units(order_idx)

                if candidate_units < self.lb:
                    continue

                candidate_demand = self._build_demand(candidate_orders)
                if not candidate_demand:
                    continue

                if not self._demand_is_coverable_by_aisles(
                    candidate_demand, cur_aisles
                ):
                    continue

                candidate_aisles = self._cleanup_redundant_aisles(
                    candidate_orders,
                    cur_aisles,
                )
                if not candidate_aisles:
                    continue

                if len(candidate_aisles) >= len(cur_aisles):
                    continue

                candidate_obj = candidate_units / len(candidate_aisles)
                if candidate_obj + eps < cur_obj:
                    continue

                cur_orders = candidate_orders
                cur_aisles = candidate_aisles
                cur_units = candidate_units
                cur_obj = candidate_obj
                improved = True
                break

            if not improved:
                break

        return cur_orders, cur_aisles, cur_units

    def _is_better_solution(
        self,
        cand_orders: list[int],
        cand_aisles: list[int],
        best_orders: list[int],
        best_aisles: list[int],
        eps: float,
    ) -> bool:
        if not cand_orders or not cand_aisles:
            return False

        cand_units = self._total_units(cand_orders)
        best_units = self._total_units(best_orders)

        cand_obj = cand_units / len(cand_aisles)
        best_obj = best_units / len(best_aisles) if best_aisles else -1.0

        if cand_obj > best_obj + eps:
            return True

        if abs(cand_obj - best_obj) <= eps and cand_units > best_units:
            return True

        return (
            abs(cand_obj - best_obj) <= eps
            and cand_units == best_units
            and (not best_aisles or len(cand_aisles) < len(best_aisles))
        )

    def solve(self) -> tuple[list[int], list[int]]:
        if self.n_orders == 0 or self.n_aisles == 0:
            return [], []

        phase1_mode = self.config.get("phase1_mode", "both")
        max_iterations = max(1, int(self.config.get("max_iterations", 25)))
        min_improvement_delta = float(self.config.get("min_improvement_delta", 1e-6))
        eps = 1e-12

        seed = self.config.get("random_seed", None)
        rng = random.Random(seed)

        if phase1_mode == "both":
            phase1_modes = ["greedy", "random"]
        elif phase1_mode in {"greedy", "random"}:
            phase1_modes = [phase1_mode]
        else:
            phase1_modes = ["greedy", "random"]

        best_orders: list[int] = []
        best_aisles: list[int] = []

        current_orders: list[int] = []
        previous_obj = -1.0

        for _ in range(max_iterations):
            round_best_orders: list[int] = []
            round_best_aisles: list[int] = []
            round_best_units = -1

            for mode in phase1_modes:
                p1_orders, p1_aisles, p1_units = self._phase1_order_to_aisle(
                    mode,
                    rng,
                    current_orders,
                )

                if not p1_orders:
                    continue

                p2_orders, p2_units = self._phase2_aisle_to_order(p1_orders, p1_aisles)
                p2_aisles = self._cleanup_redundant_aisles(p2_orders, p1_aisles)
                if not p2_aisles:
                    continue

                p3_orders, p3_aisles, p3_units = self._phase3_pruning(
                    p2_orders,
                    p2_aisles,
                    p2_units,
                    eps,
                )

                if p3_units < self.lb or not p3_aisles:
                    continue

                if not self._is_better_solution(
                    p3_orders,
                    p3_aisles,
                    round_best_orders,
                    round_best_aisles,
                    eps,
                ):
                    continue

                round_best_orders = p3_orders
                round_best_aisles = p3_aisles
                round_best_units = p3_units

            if round_best_units < self.lb or not round_best_aisles:
                current_orders = []
                continue

            round_obj = round_best_units / len(round_best_aisles)

            if self._is_better_solution(
                round_best_orders,
                round_best_aisles,
                best_orders,
                best_aisles,
                eps,
            ):
                best_orders = round_best_orders
                best_aisles = round_best_aisles

            if previous_obj >= 0.0 and round_obj - previous_obj < min_improvement_delta:
                current_orders = round_best_orders
                break

            previous_obj = round_obj
            current_orders = round_best_orders

        if not best_orders or not best_aisles:
            return self._construct_greedy_fallback()

        return best_orders, best_aisles
