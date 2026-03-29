import random

from models.solver import Solver


class TabuSearchHeuristic(Solver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)

        seed = self.config.get("random_seed")
        self.rng = random.Random(seed)

        self.max_iterations = max(1, int(self.config.get("max_iterations", 180)))
        self.no_improve_limit = max(
            1,
            int(self.config.get("no_improve_limit", self.max_iterations // 3)),
        )
        self.neighborhood_samples = max(
            1,
            int(self.config.get("neighborhood_samples", 40)),
        )
        self.tabu_tenure_orders = max(1, int(self.config.get("tabu_tenure_orders", 10)))
        self.tabu_tenure_aisles = max(1, int(self.config.get("tabu_tenure_aisles", 8)))

    def _order_units(self, order_idx: int) -> int:
        return sum(self.orders[order_idx].values())

    def _build_demand(self, selected_orders: set[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _build_inventory(self, selected_aisles: set[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in selected_aisles:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

    def _residual_demand(
        self,
        demand: dict[int, int],
        inventory: dict[int, int],
    ) -> dict[int, int]:
        residual: dict[int, int] = {}
        for item, qty in demand.items():
            missing = qty - inventory.get(item, 0)
            if missing > 0:
                residual[item] = missing
        return residual

    def _repair_aisles_for_orders(
        self,
        selected_orders: set[int],
        selected_aisles: set[int],
        cleanup: bool = True,
    ) -> set[int] | None:
        if not selected_orders:
            return set()

        demand = self._build_demand(selected_orders)
        working_aisles = set(selected_aisles)

        while True:
            inventory = self._build_inventory(working_aisles)
            residual = self._residual_demand(demand, inventory)
            if not residual:
                break

            best_aisle = None
            best_score = 0
            for aisle_idx in range(self.n_aisles):
                if aisle_idx in working_aisles:
                    continue

                score = 0
                aisle = self.aisles[aisle_idx]
                for item, needed in residual.items():
                    score += min(needed, aisle.get(item, 0))

                if score > best_score:
                    best_score = score
                    best_aisle = aisle_idx

            if best_aisle is None or best_score <= 0:
                return None

            working_aisles.add(best_aisle)

        if cleanup:
            return self._cleanup_redundant_aisles(selected_orders, working_aisles)

        return working_aisles

    def _cleanup_redundant_aisles(
        self,
        selected_orders: set[int],
        selected_aisles: set[int],
    ) -> set[int]:
        if not selected_orders:
            return set()

        demand = self._build_demand(selected_orders)
        cleaned = set(selected_aisles)

        for aisle_idx in sorted(list(cleaned)):
            if len(cleaned) == 1:
                break

            candidate = set(cleaned)
            candidate.remove(aisle_idx)
            inventory = self._build_inventory(candidate)
            if not self._residual_demand(demand, inventory):
                cleaned = candidate

        return cleaned

    def _total_units(self, selected_orders: set[int]) -> int:
        return sum(self._order_units(order_idx) for order_idx in selected_orders)

    def _try_fill_orders(
        self,
        selected_orders: set[int],
        selected_aisles: set[int],
    ) -> tuple[set[int], int]:
        units = self._total_units(selected_orders)
        inventory = self._build_inventory(selected_aisles)

        candidates = list(range(self.n_orders))
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda idx: (-self._order_units(idx), idx))

        for order_idx in candidates:
            if order_idx in selected_orders:
                continue

            order_units = self._order_units(order_idx)
            if units + order_units > self.ub:
                continue

            feasible = True
            for item, qty in self.orders[order_idx].items():
                if inventory.get(item, 0) < qty:
                    feasible = False
                    break

            if not feasible:
                continue

            selected_orders.add(order_idx)
            units += order_units
            for item, qty in self.orders[order_idx].items():
                inventory[item] = inventory.get(item, 0) - qty

        return selected_orders, units

    def _state_objective(
        self, selected_orders: set[int], selected_aisles: set[int]
    ) -> float:
        if not selected_aisles:
            return 0.0
        return self._total_units(selected_orders) / len(selected_aisles)

    def _is_state_feasible(
        self, selected_orders: set[int], selected_aisles: set[int]
    ) -> bool:
        if not selected_orders or not selected_aisles:
            return False

        total_units = self._total_units(selected_orders)
        if total_units < self.lb or total_units > self.ub:
            return False

        demand = self._build_demand(selected_orders)
        inventory = self._build_inventory(selected_aisles)
        return not self._residual_demand(demand, inventory)

    def _construct_initial_solution(self) -> tuple[set[int], set[int]]:
        init_attempts = max(1, int(self.config.get("initial_attempts", 8)))
        global_stock = self._build_inventory(set(range(self.n_aisles)))

        best_orders: set[int] = set()
        best_aisles: set[int] = set()
        best_obj = -1.0

        all_orders = list(range(self.n_orders))

        for attempt in range(init_attempts):
            remaining_stock = dict(global_stock)
            selected_orders: set[int] = set()
            units = 0

            ranked_orders = list(all_orders)
            self.rng.shuffle(ranked_orders)
            if attempt % 2 == 0:
                ranked_orders.sort(key=lambda idx: (-self._order_units(idx), idx))
            else:
                ranked_orders.sort(key=lambda idx: (self._order_units(idx), idx))

            target_units = self.rng.randint(self.lb, self.ub)

            for order_idx in ranked_orders:
                order_units = self._order_units(order_idx)
                if units + order_units > self.ub:
                    continue

                feasible = True
                for item, qty in self.orders[order_idx].items():
                    if remaining_stock.get(item, 0) < qty:
                        feasible = False
                        break

                if not feasible:
                    continue

                selected_orders.add(order_idx)
                units += order_units
                for item, qty in self.orders[order_idx].items():
                    remaining_stock[item] = remaining_stock.get(item, 0) - qty

                if units >= target_units:
                    break

            if units < self.lb:
                continue

            selected_aisles = self._repair_aisles_for_orders(selected_orders, set())
            if selected_aisles is None:
                continue

            selected_orders, _ = self._try_fill_orders(selected_orders, selected_aisles)

            selected_aisles = self._repair_aisles_for_orders(
                selected_orders,
                selected_aisles,
            )
            if selected_aisles is None:
                continue

            if not self._is_state_feasible(selected_orders, selected_aisles):
                continue

            obj = self._state_objective(selected_orders, selected_aisles)
            if obj > best_obj:
                best_obj = obj
                best_orders = set(selected_orders)
                best_aisles = set(selected_aisles)

        return best_orders, best_aisles

    def _choose_sample(self, values: list[int], sample_size: int) -> list[int]:
        if not values:
            return []
        if len(values) <= sample_size:
            return list(values)
        return self.rng.sample(values, sample_size)

    def _is_order_action_tabu(
        self,
        action: str,
        order_idx: int,
        iteration: int,
        tabu_order_add: dict[int, int],
        tabu_order_remove: dict[int, int],
    ) -> bool:
        if action == "add":
            return tabu_order_add.get(order_idx, -1) > iteration
        if action == "remove":
            return tabu_order_remove.get(order_idx, -1) > iteration
        return False

    def _is_aisle_action_tabu(
        self,
        action: str,
        aisle_idx: int,
        iteration: int,
        tabu_aisle_add: dict[int, int],
        tabu_aisle_remove: dict[int, int],
    ) -> bool:
        if action == "add":
            return tabu_aisle_add.get(aisle_idx, -1) > iteration
        if action == "remove":
            return tabu_aisle_remove.get(aisle_idx, -1) > iteration
        return False

    def solve(self) -> tuple[list[int], list[int]]:
        current_orders, current_aisles = self._construct_initial_solution()
        if not current_orders or not current_aisles:
            return [], []

        if not self._is_state_feasible(current_orders, current_aisles):
            return [], []

        best_orders = set(current_orders)
        best_aisles = set(current_aisles)
        best_obj = self._state_objective(best_orders, best_aisles)

        tabu_order_add: dict[int, int] = {}
        tabu_order_remove: dict[int, int] = {}
        tabu_aisle_add: dict[int, int] = {}
        tabu_aisle_remove: dict[int, int] = {}

        no_improve = 0

        for iteration in range(1, self.max_iterations + 1):
            candidate_best = None
            candidate_obj = -1.0
            candidate_move = None

            selected_list = list(current_orders)
            non_selected_list = [
                idx for idx in range(self.n_orders) if idx not in current_orders
            ]
            current_units = self._total_units(current_orders)

            sampled_selected = self._choose_sample(
                selected_list,
                min(len(selected_list), self.neighborhood_samples),
            )
            sampled_non_selected = self._choose_sample(
                non_selected_list,
                min(len(non_selected_list), self.neighborhood_samples),
            )

            for order_in in sampled_non_selected:
                order_units = self._order_units(order_in)
                if current_units + order_units > self.ub:
                    continue

                candidate_orders = set(current_orders)
                candidate_orders.add(order_in)
                candidate_aisles = self._repair_aisles_for_orders(
                    candidate_orders,
                    set(current_aisles),
                    cleanup=False,
                )

                if candidate_aisles is None:
                    continue

                if not self._is_state_feasible(candidate_orders, candidate_aisles):
                    continue

                obj = self._state_objective(candidate_orders, candidate_aisles)
                tabu = self._is_order_action_tabu(
                    "add",
                    order_in,
                    iteration,
                    tabu_order_add,
                    tabu_order_remove,
                )

                if tabu and obj <= best_obj:
                    continue

                if obj > candidate_obj:
                    candidate_obj = obj
                    candidate_best = (candidate_orders, candidate_aisles)
                    candidate_move = ("order_add", order_in)

            for order_out in sampled_selected:
                candidate_orders = set(current_orders)
                candidate_orders.remove(order_out)

                if self._total_units(candidate_orders) < self.lb:
                    continue

                candidate_aisles = self._repair_aisles_for_orders(
                    candidate_orders,
                    set(current_aisles),
                    cleanup=False,
                )
                if candidate_aisles is None:
                    continue

                if not self._is_state_feasible(candidate_orders, candidate_aisles):
                    continue

                obj = self._state_objective(candidate_orders, candidate_aisles)
                tabu = self._is_order_action_tabu(
                    "remove",
                    order_out,
                    iteration,
                    tabu_order_add,
                    tabu_order_remove,
                )

                if tabu and obj <= best_obj:
                    continue

                if obj > candidate_obj:
                    candidate_obj = obj
                    candidate_best = (candidate_orders, candidate_aisles)
                    candidate_move = ("order_remove", order_out)

            if sampled_selected and sampled_non_selected:
                swap_sample = min(
                    len(sampled_selected) * len(sampled_non_selected),
                    self.neighborhood_samples,
                )

                for _ in range(swap_sample):
                    order_out = self.rng.choice(sampled_selected)
                    order_in = self.rng.choice(sampled_non_selected)

                    if order_in == order_out:
                        continue

                    candidate_orders = set(current_orders)
                    candidate_orders.remove(order_out)
                    candidate_orders.add(order_in)

                    units = self._total_units(candidate_orders)
                    if units < self.lb or units > self.ub:
                        continue

                    candidate_aisles = self._repair_aisles_for_orders(
                        candidate_orders,
                        set(current_aisles),
                        cleanup=False,
                    )
                    if candidate_aisles is None:
                        continue

                    if not self._is_state_feasible(candidate_orders, candidate_aisles):
                        continue

                    obj = self._state_objective(candidate_orders, candidate_aisles)
                    tabu_add = self._is_order_action_tabu(
                        "add",
                        order_in,
                        iteration,
                        tabu_order_add,
                        tabu_order_remove,
                    )
                    tabu_remove = self._is_order_action_tabu(
                        "remove",
                        order_out,
                        iteration,
                        tabu_order_add,
                        tabu_order_remove,
                    )

                    if (tabu_add or tabu_remove) and obj <= best_obj:
                        continue

                    if obj > candidate_obj:
                        candidate_obj = obj
                        candidate_best = (candidate_orders, candidate_aisles)
                        candidate_move = ("order_swap", order_out, order_in)

            current_aisles_list = list(current_aisles)
            non_current_aisles_list = [
                idx for idx in range(self.n_aisles) if idx not in current_aisles
            ]

            sampled_current_aisles = self._choose_sample(
                current_aisles_list,
                min(len(current_aisles_list), self.neighborhood_samples),
            )
            sampled_non_current_aisles = self._choose_sample(
                non_current_aisles_list,
                min(len(non_current_aisles_list), self.neighborhood_samples),
            )

            for aisle_out in sampled_current_aisles:
                candidate_aisles_seed = set(current_aisles)
                candidate_aisles_seed.remove(aisle_out)

                candidate_aisles = self._repair_aisles_for_orders(
                    set(current_orders),
                    candidate_aisles_seed,
                    cleanup=False,
                )
                if candidate_aisles is None:
                    continue

                if not self._is_state_feasible(current_orders, candidate_aisles):
                    continue

                obj = self._state_objective(current_orders, candidate_aisles)
                tabu = self._is_aisle_action_tabu(
                    "remove",
                    aisle_out,
                    iteration,
                    tabu_aisle_add,
                    tabu_aisle_remove,
                )

                if tabu and obj <= best_obj:
                    continue

                if obj > candidate_obj:
                    candidate_obj = obj
                    candidate_best = (set(current_orders), candidate_aisles)
                    candidate_move = ("aisle_remove", aisle_out)

            if sampled_current_aisles and sampled_non_current_aisles:
                swap_sample = min(
                    len(sampled_current_aisles) * len(sampled_non_current_aisles),
                    self.neighborhood_samples,
                )

                for _ in range(swap_sample):
                    aisle_out = self.rng.choice(sampled_current_aisles)
                    aisle_in = self.rng.choice(sampled_non_current_aisles)

                    candidate_aisles_seed = set(current_aisles)
                    candidate_aisles_seed.remove(aisle_out)
                    candidate_aisles_seed.add(aisle_in)

                    candidate_aisles = self._repair_aisles_for_orders(
                        set(current_orders),
                        candidate_aisles_seed,
                        cleanup=False,
                    )

                    if candidate_aisles is None:
                        continue

                    if not self._is_state_feasible(current_orders, candidate_aisles):
                        continue

                    obj = self._state_objective(current_orders, candidate_aisles)
                    tabu_add = self._is_aisle_action_tabu(
                        "add",
                        aisle_in,
                        iteration,
                        tabu_aisle_add,
                        tabu_aisle_remove,
                    )
                    tabu_remove = self._is_aisle_action_tabu(
                        "remove",
                        aisle_out,
                        iteration,
                        tabu_aisle_add,
                        tabu_aisle_remove,
                    )

                    if (tabu_add or tabu_remove) and obj <= best_obj:
                        continue

                    if obj > candidate_obj:
                        candidate_obj = obj
                        candidate_best = (set(current_orders), candidate_aisles)
                        candidate_move = ("aisle_swap", aisle_out, aisle_in)

            if candidate_best is None:
                break

            current_orders, current_aisles = candidate_best
            current_aisles = self._cleanup_redundant_aisles(
                current_orders, current_aisles
            )

            if candidate_move is not None:
                move_type = candidate_move[0]
                if move_type == "order_add":
                    order_idx = candidate_move[1]
                    tabu_order_remove[order_idx] = iteration + self.tabu_tenure_orders
                elif move_type == "order_remove":
                    order_idx = candidate_move[1]
                    tabu_order_add[order_idx] = iteration + self.tabu_tenure_orders
                elif move_type == "order_swap":
                    removed_order = candidate_move[1]
                    added_order = candidate_move[2]
                    tabu_order_add[removed_order] = iteration + self.tabu_tenure_orders
                    tabu_order_remove[added_order] = iteration + self.tabu_tenure_orders
                elif move_type == "aisle_add":
                    aisle_idx = candidate_move[1]
                    tabu_aisle_remove[aisle_idx] = iteration + self.tabu_tenure_aisles
                elif move_type == "aisle_remove":
                    aisle_idx = candidate_move[1]
                    tabu_aisle_add[aisle_idx] = iteration + self.tabu_tenure_aisles
                elif move_type == "aisle_swap":
                    removed_aisle = candidate_move[1]
                    added_aisle = candidate_move[2]
                    tabu_aisle_add[removed_aisle] = iteration + self.tabu_tenure_aisles
                    tabu_aisle_remove[added_aisle] = iteration + self.tabu_tenure_aisles

            current_obj = self._state_objective(current_orders, current_aisles)

            if current_obj > best_obj:
                best_obj = current_obj
                best_orders = set(current_orders)
                best_aisles = set(current_aisles)
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.no_improve_limit:
                current_orders, current_aisles = self._construct_initial_solution()
                if not current_orders or not current_aisles:
                    no_improve = 0
                    continue
                no_improve = 0

        if not self._is_state_feasible(best_orders, best_aisles):
            return [], []

        return sorted(best_orders), sorted(best_aisles)
