import math
import random
import time

from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from models.solver import Solver


class SimulatedAnnealingHeuristic(Solver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)

        self.max_iterations = max(1, int(self.config.get("max_iterations", 4000)))
        self.max_time_seconds = float(self.config.get("max_time_seconds", 1.5))

        self.initial_temp = max(1e-9, float(self.config.get("initial_temp", 2.5)))
        self.cooling_rate = float(self.config.get("cooling_rate", 0.9975))
        self.min_temp = max(1e-9, float(self.config.get("min_temp", 1e-4)))

        self.neighbor_tries = max(1, int(self.config.get("neighbor_tries", 40)))
        self.initial_attempts = max(1, int(self.config.get("initial_attempts", 24)))

        self.order_units = [sum(order.values()) for order in self.orders]

    def _build_demand(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _build_supply(self, aisle_indices: list[int]) -> dict[int, int]:
        supply: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                supply[item] = supply.get(item, 0) + qty
        return supply

    def _is_demand_coverable(
        self,
        demand: dict[int, int],
        aisle_indices: list[int],
    ) -> bool:
        supply = self._build_supply(aisle_indices)
        for item, qty in demand.items():
            if supply.get(item, 0) < qty:
                return False
        return True

    def _cleanup_redundant_aisles(
        self,
        selected_orders: list[int],
        aisle_indices: list[int],
    ) -> list[int]:
        if not selected_orders:
            return []

        demand = self._build_demand(selected_orders)
        cleaned = sorted(set(aisle_indices))

        for aisle_idx in list(cleaned):
            if len(cleaned) == 1:
                break

            candidate = [idx for idx in cleaned if idx != aisle_idx]
            if self._is_demand_coverable(demand, candidate):
                cleaned = candidate

        return cleaned

    def _total_units(self, selected_orders: list[int]) -> int:
        return sum(self.order_units[order_idx] for order_idx in selected_orders)

    def _is_order_coverable_by_global_stock(
        self, order_idx: int, stock: dict[int, int]
    ) -> bool:
        for item, qty in self.orders[order_idx].items():
            if stock.get(item, 0) < qty:
                return False
        return True

    def _evaluate_solution(self, selected_orders: list[int]):
        if not selected_orders:
            return None

        unique_orders = sorted(set(selected_orders))
        units = self._total_units(unique_orders)

        if units < self.lb or units > self.ub:
            return None

        demand = self._build_demand(unique_orders)
        visited_aisles = multi_greedy_aisle_select(demand, self.aisles)
        if not visited_aisles:
            return None

        if not self._is_demand_coverable(demand, visited_aisles):
            return None

        cleaned_aisles = self._cleanup_redundant_aisles(unique_orders, visited_aisles)
        if not cleaned_aisles:
            return None

        objective = units / len(cleaned_aisles)

        return {
            "orders": unique_orders,
            "aisles": cleaned_aisles,
            "units": units,
            "objective": objective,
        }

    def _is_better(self, candidate, incumbent) -> bool:
        if incumbent is None:
            return True

        if candidate["objective"] > incumbent["objective"]:
            return True
        if candidate["objective"] < incumbent["objective"]:
            return False

        if len(candidate["aisles"]) < len(incumbent["aisles"]):
            return True
        if len(candidate["aisles"]) > len(incumbent["aisles"]):
            return False

        if candidate["units"] > incumbent["units"]:
            return True
        if candidate["units"] < incumbent["units"]:
            return False

        return len(candidate["orders"]) > len(incumbent["orders"])

    def _build_initial_solution(self, rng: random.Random):
        global_stock: dict[int, int] = {}
        for aisle in self.aisles:
            for item, qty in aisle.items():
                global_stock[item] = global_stock.get(item, 0) + qty

        ranked_orders = sorted(
            range(self.n_orders),
            key=lambda idx: (-self.order_units[idx], idx),
        )

        best = None

        for attempt in range(self.initial_attempts):
            if attempt == 0:
                sequence = list(ranked_orders)
            else:
                sequence = list(ranked_orders)
                rng.shuffle(sequence)

            remaining_stock = dict(global_stock)
            selected_orders: list[int] = []
            units = 0

            for order_idx in sequence:
                order_units = self.order_units[order_idx]
                if units + order_units > self.ub:
                    continue

                if not self._is_order_coverable_by_global_stock(
                    order_idx, remaining_stock
                ):
                    continue

                selected_orders.append(order_idx)
                units += order_units

                for item, qty in self.orders[order_idx].items():
                    remaining_stock[item] = remaining_stock.get(item, 0) - qty

            evaluated = self._evaluate_solution(selected_orders)
            if evaluated is None:
                continue

            if self._is_better(evaluated, best):
                best = evaluated

        return best

    def _add_neighbor(
        self, current_orders: list[int], current_units: int, rng: random.Random
    ):
        selected_set = set(current_orders)
        candidates = [
            idx
            for idx in range(self.n_orders)
            if idx not in selected_set
            and current_units + self.order_units[idx] <= self.ub
        ]

        if not candidates:
            return None

        order_to_add = rng.choice(candidates)
        new_orders = list(current_orders)
        new_orders.append(order_to_add)
        return self._evaluate_solution(new_orders)

    def _remove_neighbor(
        self, current_orders: list[int], current_units: int, rng: random.Random
    ):
        removable = [
            idx
            for idx in current_orders
            if current_units - self.order_units[idx] >= self.lb
        ]

        if not removable:
            return None

        order_to_remove = rng.choice(removable)
        new_orders = [idx for idx in current_orders if idx != order_to_remove]
        return self._evaluate_solution(new_orders)

    def _swap_neighbor(
        self, current_orders: list[int], current_units: int, rng: random.Random
    ):
        if not current_orders:
            return None

        selected_set = set(current_orders)
        outside_orders = [
            idx for idx in range(self.n_orders) if idx not in selected_set
        ]
        if not outside_orders:
            return None

        inside = list(current_orders)
        rng.shuffle(inside)
        outside = list(outside_orders)
        rng.shuffle(outside)

        for in_order in inside:
            for out_order in outside:
                new_units = (
                    current_units
                    - self.order_units[in_order]
                    + self.order_units[out_order]
                )
                if new_units < self.lb or new_units > self.ub:
                    continue

                candidate_orders = [idx for idx in current_orders if idx != in_order]
                candidate_orders.append(out_order)
                evaluated = self._evaluate_solution(candidate_orders)
                if evaluated is not None:
                    return evaluated

        return None

    def _propose_neighbor(self, current, rng: random.Random):
        current_orders = current["orders"]
        current_units = current["units"]
        selected_set = set(current_orders)

        can_add = any(
            idx not in selected_set and current_units + self.order_units[idx] <= self.ub
            for idx in range(self.n_orders)
        )
        can_remove = any(
            current_units - self.order_units[idx] >= self.lb for idx in current_orders
        )
        can_swap = len(current_orders) > 0 and len(selected_set) < self.n_orders

        available_moves = []
        if can_add:
            available_moves.append("add")
        if can_remove:
            available_moves.append("remove")
        if can_swap:
            available_moves.append("swap")

        if not available_moves:
            return None

        for _ in range(self.neighbor_tries):
            move = rng.choice(available_moves)

            if move == "add":
                candidate = self._add_neighbor(current_orders, current_units, rng)
            elif move == "remove":
                candidate = self._remove_neighbor(current_orders, current_units, rng)
            else:
                candidate = self._swap_neighbor(current_orders, current_units, rng)

            if candidate is not None:
                return candidate

        return None

    def solve(self) -> tuple[list[int], list[int]]:
        if self.n_orders == 0 or self.n_aisles == 0:
            return [], []

        seed = self.config.get("random_seed", self.config.get("seed"))
        rng = random.Random(seed)

        current = self._build_initial_solution(rng)
        if current is None:
            return [], []

        best = current
        temperature = self.initial_temp
        start_time = time.perf_counter()

        for _ in range(self.max_iterations):
            elapsed = time.perf_counter() - start_time
            if elapsed >= self.max_time_seconds:
                break

            neighbor = self._propose_neighbor(current, rng)
            if neighbor is None:
                temperature = max(self.min_temp, temperature * self.cooling_rate)
                continue

            delta = neighbor["objective"] - current["objective"]
            accept = False

            if delta >= 0:
                accept = True
            elif temperature > 0.0:
                acceptance_prob = math.exp(delta / max(temperature, self.min_temp))
                if rng.random() < acceptance_prob:
                    accept = True

            if accept:
                current = neighbor
                if self._is_better(current, best):
                    best = current

            temperature = max(self.min_temp, temperature * self.cooling_rate)

        return best["orders"], best["aisles"]
