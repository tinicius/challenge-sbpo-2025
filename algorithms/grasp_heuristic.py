import random

from algorithms.base import Algorithm
from problems.base import ProblemInput


class GRASPHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "grasp_heuristic"

    def _order_units(self) -> list[int]:
        return [sum(order.values()) for order in self.orders]

    def _build_total_demand(self) -> dict[int, int]:
        total_demand: dict[int, int] = {}
        for order in self.orders:
            for item, qty in order.items():
                total_demand[item] = total_demand.get(item, 0) + qty
        return total_demand

    def _inventory_for_aisles(self, aisle_indexes: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indexes:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _is_order_coverable(self, order_idx: int, inventory: dict[int, int]) -> bool:
        for item, qty in self.orders[order_idx].items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _pack_orders_in_sequence(
        self,
        order_sequence: list[int],
        inventory: dict[int, int],
        order_units: list[int],
    ) -> tuple[list[int], int]:
        remaining_inventory = dict(inventory)
        selected_orders: list[int] = []
        total_units = 0

        for order_idx in order_sequence:
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

    def _best_orders_for_inventory(
        self,
        inventory: dict[int, int],
        order_units: list[int],
        rng: random.Random,
    ) -> tuple[list[int], int]:
        candidate_orders = [
            order_idx
            for order_idx in range(self.n_orders)
            if self._is_order_coverable(order_idx, inventory)
        ]

        if not candidate_orders:
            return [], 0

        sequences: list[list[int]] = []
        sequences.append(
            sorted(candidate_orders, key=lambda idx: (-order_units[idx], idx))
        )
        sequences.append(
            sorted(candidate_orders, key=lambda idx: (order_units[idx], idx))
        )

        shuffled = list(candidate_orders)
        rng.shuffle(shuffled)
        sequences.append(shuffled)

        best_orders: list[int] = []
        best_units = -1

        for sequence in sequences:
            packed_orders, packed_units = self._pack_orders_in_sequence(
                sequence,
                inventory,
                order_units,
            )

            if packed_units > best_units:
                best_orders = packed_orders
                best_units = packed_units

        if best_units < 0:
            return [], 0

        return best_orders, best_units

    def _score_candidate_aisles(
        self,
        selected_aisles: list[int],
        total_demand: dict[int, int],
    ) -> list[tuple[int, int]]:
        current_inventory = self._inventory_for_aisles(selected_aisles)
        remaining_coverage: dict[int, int] = {}

        for item, qty in total_demand.items():
            covered = min(qty, current_inventory.get(item, 0))
            remaining_coverage[item] = qty - covered

        scored: list[tuple[int, int]] = []

        for aisle_idx in range(self.n_aisles):
            if aisle_idx in selected_aisles:
                continue

            score = 0
            aisle = self.aisles[aisle_idx]
            for item, qty in aisle.items():
                score += min(qty, remaining_coverage.get(item, 0))

            if score > 0:
                scored.append((aisle_idx, score))

        scored.sort(key=lambda value: (-value[1], value[0]))
        return scored

    def _build_rcl(
        self, scored_aisles: list[tuple[int, int]], alpha: float
    ) -> list[int]:
        if not scored_aisles:
            return []

        best_score = scored_aisles[0][1]
        worst_score = scored_aisles[-1][1]
        threshold = best_score - alpha * (best_score - worst_score)

        return [aisle_idx for aisle_idx, score in scored_aisles if score >= threshold]

    def _construct_solution(
        self,
        alpha: float,
        order_units: list[int],
        total_demand: dict[int, int],
        rng: random.Random,
    ) -> tuple[list[int], list[int], int]:
        selected_aisles: list[int] = []
        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_units = 0
        best_obj = -1.0
        progress_units = 0

        max_aisles_in_construction = int(
            self.config.get("max_aisles_in_construction", self.n_aisles)
        )
        max_aisles_in_construction = max(
            1,
            min(self.n_aisles, max_aisles_in_construction),
        )

        stagnation_limit = int(self.config.get("construction_stagnation", 4))
        stagnation_limit = max(1, stagnation_limit)
        stagnation = 0

        while (
            len(selected_aisles) < max_aisles_in_construction
            and stagnation < stagnation_limit
        ):
            scored_aisles = self._score_candidate_aisles(selected_aisles, total_demand)
            if not scored_aisles:
                break

            rcl = self._build_rcl(scored_aisles, alpha)
            if not rcl:
                break

            chosen_aisle = rng.choice(rcl)
            selected_aisles.append(chosen_aisle)

            inventory = self._inventory_for_aisles(selected_aisles)
            candidate_orders, candidate_units = self._best_orders_for_inventory(
                inventory,
                order_units,
                rng,
            )

            improved = False
            if candidate_units >= self.lb and selected_aisles:
                candidate_obj = candidate_units / len(selected_aisles)
                if candidate_obj > best_obj:
                    improved = True
                elif candidate_obj == best_obj and len(selected_aisles) < len(
                    best_aisles
                ):
                    improved = True
                elif (
                    candidate_obj == best_obj
                    and len(selected_aisles) == len(best_aisles)
                    and candidate_units > best_units
                ):
                    improved = True

                if improved:
                    best_obj = candidate_obj
                    best_units = candidate_units
                    best_orders = list(candidate_orders)
                    best_aisles = sorted(selected_aisles)
            elif candidate_units > progress_units:
                # Keep growing the constructive phase while we are still progressing toward LB.
                progress_units = candidate_units
                improved = True

            if improved:
                stagnation = 0
            else:
                stagnation += 1

        return best_orders, best_aisles, best_units

    def _construct_greedy_fallback(
        self, order_units: list[int]
    ) -> tuple[list[int], list[int], int]:
        global_stock: dict[int, int] = {}
        for aisle in self.aisles:
            for item, qty in aisle.items():
                global_stock[item] = global_stock.get(item, 0) + qty

        selected_orders: list[int] = []
        total_units = 0

        ranked_orders = sorted(
            range(self.n_orders),
            key=lambda idx: (-order_units[idx], idx),
        )

        for order_idx in ranked_orders:
            order = self.orders[order_idx]
            units = order_units[order_idx]

            if total_units + units > self.ub:
                continue

            feasible = True
            for item, qty in order.items():
                if global_stock.get(item, 0) < qty:
                    feasible = False
                    break

            if not feasible:
                continue

            selected_orders.append(order_idx)
            total_units += units

            for item, qty in order.items():
                global_stock[item] = global_stock.get(item, 0) - qty

        if total_units < self.lb:
            return [], [], 0

        demand = self._build_demand_from_orders(selected_orders)
        selected_aisles = []
        remaining_demand = dict(demand)
        available_aisles = set(range(self.n_aisles))

        while remaining_demand and available_aisles:
            best_aisle = -1
            best_score = 0

            for aisle_idx in available_aisles:
                score = sum(
                    min(remaining_demand.get(item, 0), qty)
                    for item, qty in self.aisles[aisle_idx].items()
                )
                if score > best_score:
                    best_score = score
                    best_aisle = aisle_idx

            if best_score == 0:
                break

            selected_aisles.append(best_aisle)
            available_aisles.remove(best_aisle)

            for item, qty in self.aisles[best_aisle].items():
                if item in remaining_demand:
                    remaining_demand[item] -= qty
                    if remaining_demand[item] <= 0:
                        del remaining_demand[item]

        if remaining_demand:
            return [], [], 0

        cleaned = self._cleanup_redundant_aisles(selected_orders, selected_aisles)
        if not cleaned:
            return [], [], 0

        return selected_orders, cleaned, total_units

    def _build_demand_from_orders(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _is_demand_coverable(
        self,
        demand: dict[int, int],
        aisle_indices: list[int],
    ) -> bool:
        inventory = self._inventory_for_aisles(aisle_indices)
        for item, qty in demand.items():
            if inventory.get(item, 0) < qty:
                return False
        return True

    def _cleanup_redundant_aisles(
        self, selected_orders: list[int], aisle_indices: list[int]
    ) -> list[int]:
        if not selected_orders:
            return []

        demand = self._build_demand_from_orders(selected_orders)
        cleaned = sorted(set(aisle_indices))

        for aisle_idx in list(cleaned):
            if len(cleaned) == 1:
                break

            candidate = [idx for idx in cleaned if idx != aisle_idx]
            if self._is_demand_coverable(demand, candidate):
                cleaned = candidate

        return cleaned

    def _compare_solutions(
        self,
        orders_a: list[int],
        aisles_a: list[int],
        units_a: int,
        orders_b: list[int],
        aisles_b: list[int],
        units_b: int,
    ) -> int:
        if not aisles_a:
            return -1
        if not aisles_b:
            return 1

        obj_a = units_a / len(aisles_a)
        obj_b = units_b / len(aisles_b)

        if obj_a > obj_b:
            return 1
        if obj_a < obj_b:
            return -1

        if len(aisles_a) < len(aisles_b):
            return 1
        if len(aisles_a) > len(aisles_b):
            return -1

        if units_a > units_b:
            return 1
        if units_a < units_b:
            return -1

        if len(orders_a) > len(orders_b):
            return 1
        if len(orders_a) < len(orders_b):
            return -1

        return 0

    def _compute_units(self, selected_orders: list[int], order_units: list[int]) -> int:
        return sum(order_units[order_idx] for order_idx in selected_orders)

    def _local_search_swap_aisle(
        self,
        selected_orders: list[int],
        selected_aisles: list[int],
        order_units: list[int],
        rng: random.Random,
    ) -> tuple[list[int], list[int], int, bool]:
        if len(selected_aisles) < 2:
            return (
                selected_orders,
                selected_aisles,
                self._compute_units(selected_orders, order_units),
                False,
            )

        max_aisle_swap_checks = int(self.config.get("max_aisle_swap_checks", 5000))
        max_aisle_swap_checks = max(1, max_aisle_swap_checks)
        aisle_swap_checks = 0

        baseline_units = self._compute_units(selected_orders, order_units)
        baseline_orders = list(selected_orders)
        baseline_aisles = list(selected_aisles)

        shuffled_out = list(selected_aisles)
        shuffled_in = [
            idx for idx in range(self.n_aisles) if idx not in selected_aisles
        ]
        rng.shuffle(shuffled_out)
        rng.shuffle(shuffled_in)

        for out_aisle in shuffled_out:
            base = [idx for idx in selected_aisles if idx != out_aisle]

            inventory_base = self._inventory_for_aisles(base)
            base_orders, base_units = self._best_orders_for_inventory(
                inventory_base,
                order_units,
                rng,
            )
            if base_units >= self.lb:
                base_cleaned = self._cleanup_redundant_aisles(base_orders, base)
                if base_cleaned:
                    cmp_drop = self._compare_solutions(
                        base_orders,
                        base_cleaned,
                        base_units,
                        baseline_orders,
                        baseline_aisles,
                        baseline_units,
                    )
                    if cmp_drop > 0:
                        return base_orders, base_cleaned, base_units, True

            for in_aisle in shuffled_in:
                aisle_swap_checks += 1
                if aisle_swap_checks > max_aisle_swap_checks:
                    return baseline_orders, baseline_aisles, baseline_units, False

                candidate_aisles = sorted(base + [in_aisle])
                inventory = self._inventory_for_aisles(candidate_aisles)
                candidate_orders, candidate_units = self._best_orders_for_inventory(
                    inventory,
                    order_units,
                    rng,
                )

                if candidate_units < self.lb:
                    continue

                cleaned = self._cleanup_redundant_aisles(
                    candidate_orders, candidate_aisles
                )
                if not cleaned:
                    continue

                cmp_result = self._compare_solutions(
                    candidate_orders,
                    cleaned,
                    candidate_units,
                    baseline_orders,
                    baseline_aisles,
                    baseline_units,
                )
                if cmp_result > 0:
                    return candidate_orders, cleaned, candidate_units, True

        return baseline_orders, baseline_aisles, baseline_units, False

    def _try_order_swap_with_fixed_aisles(
        self,
        selected_orders: list[int],
        selected_aisles: list[int],
        order_units: list[int],
        rng: random.Random,
    ) -> tuple[list[int], int, bool]:
        if not selected_orders:
            return selected_orders, 0, False

        max_order_swap_checks = int(self.config.get("max_order_swap_checks", 20000))
        max_order_swap_checks = max(1, max_order_swap_checks)
        order_swap_checks = 0

        current_units = self._compute_units(selected_orders, order_units)
        best_orders = list(selected_orders)
        best_units = current_units

        inventory = self._inventory_for_aisles(selected_aisles)
        selected_set = set(selected_orders)

        consumed: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                consumed[item] = consumed.get(item, 0) + qty

        outside_orders = [
            order_idx
            for order_idx in range(self.n_orders)
            if order_idx not in selected_set
        ]

        rng.shuffle(outside_orders)
        inside_orders = list(selected_orders)
        rng.shuffle(inside_orders)

        for out_order in outside_orders:
            out_units = order_units[out_order]
            if out_units <= 0:
                continue

            for in_order in inside_orders:
                order_swap_checks += 1
                if order_swap_checks > max_order_swap_checks:
                    return selected_orders, current_units, False

                in_units = order_units[in_order]
                new_units = current_units - in_units + out_units

                if new_units > self.ub or new_units < self.lb:
                    continue

                feasible = True
                for item, qty in self.orders[out_order].items():
                    used_without_in = consumed.get(item, 0) - self.orders[in_order].get(
                        item, 0
                    )
                    if used_without_in + qty > inventory.get(item, 0):
                        feasible = False
                        break

                if not feasible:
                    continue

                candidate_orders = [idx for idx in selected_orders if idx != in_order]
                candidate_orders.append(out_order)
                candidate_orders.sort()

                if new_units > best_units:
                    best_orders = candidate_orders
                    best_units = new_units

        if best_units > current_units:
            return best_orders, best_units, True

        return selected_orders, current_units, False

    def _local_search(
        self,
        selected_orders: list[int],
        selected_aisles: list[int],
        order_units: list[int],
        rng: random.Random,
    ) -> tuple[list[int], list[int], int]:
        if not selected_orders or not selected_aisles:
            return [], [], 0

        current_orders = list(selected_orders)
        current_aisles = sorted(set(selected_aisles))
        current_units = self._compute_units(current_orders, order_units)

        max_no_improve = int(self.config.get("local_search_no_improve", 8))
        max_no_improve = max(1, max_no_improve)
        no_improve = 0

        while no_improve < max_no_improve:
            improved = False

            current_orders, current_aisles, current_units, improved_aisle = (
                self._local_search_swap_aisle(
                    current_orders,
                    current_aisles,
                    order_units,
                    rng,
                )
            )

            if improved_aisle:
                improved = True

            swapped_orders, swapped_units, improved_order = (
                self._try_order_swap_with_fixed_aisles(
                    current_orders,
                    current_aisles,
                    order_units,
                    rng,
                )
            )

            if improved_order:
                current_orders = swapped_orders
                current_units = swapped_units
                cleaned = self._cleanup_redundant_aisles(current_orders, current_aisles)
                if cleaned:
                    current_aisles = cleaned
                improved = True

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return current_orders, current_aisles, current_units

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        if self.n_orders == 0 or self.n_aisles == 0:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        max_iterations = int(self.config.get("max_iterations", 50))
        max_iterations = max(1, max_iterations)

        alpha = float(self.config.get("alpha", 0.3))
        alpha = max(0.0, min(1.0, alpha))

        random_seed = self.config.get("random_seed")
        rng = random.Random(random_seed)

        order_units = self._order_units()
        total_demand = self._build_total_demand()

        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_units = 0

        for _ in range(max_iterations):
            candidate_orders, candidate_aisles, candidate_units = (
                self._construct_solution(
                    alpha,
                    order_units,
                    total_demand,
                    rng,
                )
            )

            if candidate_units < self.lb or not candidate_aisles:
                continue

            refined_orders, refined_aisles, refined_units = self._local_search(
                candidate_orders,
                candidate_aisles,
                order_units,
                rng,
            )

            if refined_units < self.lb or not refined_aisles:
                continue

            if (
                self._compare_solutions(
                    refined_orders,
                    refined_aisles,
                    refined_units,
                    best_orders,
                    best_aisles,
                    best_units,
                )
                > 0
            ):
                best_orders = refined_orders
                best_aisles = refined_aisles
                best_units = refined_units

        if best_units < self.lb or not best_aisles:
            fallback_orders, fallback_aisles, fallback_units = (
                self._construct_greedy_fallback(order_units)
            )
            if fallback_units < self.lb or not fallback_aisles:
                return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

            selected_orders = sorted(fallback_orders)
            visited_aisles = sorted(fallback_aisles)

            if not selected_orders or not visited_aisles:
                return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

            total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
            objective = total_items / len(visited_aisles)
            return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}

        selected_orders = sorted(best_orders)
        visited_aisles = sorted(best_aisles)

        if not selected_orders or not visited_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}
