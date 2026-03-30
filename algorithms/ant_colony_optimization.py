import random
from itertools import combinations

from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class AntColonyOptimizationHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "ant_colony_optimization"

    def _build_total_stock(self) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in self.aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock

    def _compute_coverable_orders(self) -> list[int]:
        coverable: list[int] = []
        for order_idx, order in enumerate(self.orders):
            if self.order_units[order_idx] > self.ub:
                continue

            feasible = True
            for item, qty in order.items():
                if self.total_stock.get(item, 0) < qty:
                    feasible = False
                    break

            if feasible:
                coverable.append(order_idx)

        return coverable

    def _build_demand(self, selected_orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for order_idx in selected_orders:
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _inventory_for_aisles(self, aisle_indices: list[int]) -> dict[int, int]:
        inventory: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty
        return inventory

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

    def _heuristic_value(
        self,
        order_idx: int,
        current_demand: dict[int, int],
        total_units: int,
    ) -> float:
        order = self.orders[order_idx]
        units = self.order_units[order_idx]

        overlap_units = 0
        rarity_penalty = 0.0

        for item, qty in order.items():
            overlap_units += min(qty, current_demand.get(item, 0))
            if item not in current_demand:
                rarity_penalty += qty * self.item_rarity.get(item, 1.0)

        lb_progress = min(self.lb, total_units + units) / max(1, self.lb)

        numerator = units + 1.25 * overlap_units + 0.25 * lb_progress * units
        denominator = 1.0 + rarity_penalty

        return max(1e-9, numerator / denominator)

    def _pair_pheromone(
        self,
        order_idx: int,
        selected_orders: list[int],
        pheromone: list[list[float]],
    ) -> float:
        if not selected_orders:
            return self.initial_pheromone

        total = 0.0
        for other in selected_orders:
            i = min(order_idx, other)
            j = max(order_idx, other)
            total += pheromone[i][j]

        return max(1e-9, total / len(selected_orders))

    def _choose_next_order(
        self,
        candidates: list[int],
        selected_orders: list[int],
        current_demand: dict[int, int],
        total_units: int,
        pheromone: list[list[float]],
        rng: random.Random,
    ) -> int:
        scored: list[tuple[int, float, float]] = []

        for order_idx in candidates:
            tau = self._pair_pheromone(order_idx, selected_orders, pheromone)
            eta = self._heuristic_value(order_idx, current_demand, total_units)
            scored.append((order_idx, tau, eta))

        if self.candidate_list_size > 0 and len(scored) > self.candidate_list_size:
            scored.sort(key=lambda value: value[2], reverse=True)
            scored = scored[: self.candidate_list_size]

        weights: list[float] = []
        filtered_orders: list[int] = []

        for order_idx, tau, eta in scored:
            value = (tau**self.alpha) * (eta**self.beta)
            if value > 0:
                filtered_orders.append(order_idx)
                weights.append(value)

        if not filtered_orders:
            return rng.choice(candidates)

        return rng.choices(filtered_orders, weights=weights, k=1)[0]

    def _construct_ant_solution(
        self,
        pheromone: list[list[float]],
        rng: random.Random,
    ) -> list[int]:
        if not self.coverable_orders:
            return []

        available_orders = set(self.coverable_orders)
        selected_orders: list[int] = []

        total_units = 0
        remaining_stock = dict(self.total_stock)
        current_demand: dict[int, int] = {}

        while available_orders:
            candidates: list[int] = []

            for order_idx in available_orders:
                units = self.order_units[order_idx]
                if total_units + units > self.ub:
                    continue

                feasible = True
                for item, qty in self.orders[order_idx].items():
                    if remaining_stock.get(item, 0) < qty:
                        feasible = False
                        break

                if feasible:
                    candidates.append(order_idx)

            if not candidates:
                break

            chosen = self._choose_next_order(
                candidates=candidates,
                selected_orders=selected_orders,
                current_demand=current_demand,
                total_units=total_units,
                pheromone=pheromone,
                rng=rng,
            )

            selected_orders.append(chosen)
            available_orders.remove(chosen)

            order = self.orders[chosen]
            total_units += self.order_units[chosen]

            for item, qty in order.items():
                current_demand[item] = current_demand.get(item, 0) + qty
                remaining_stock[item] = remaining_stock.get(item, 0) - qty

            if total_units >= self.ub:
                break

            if total_units >= self.lb and rng.random() < self.stop_after_lb_prob:
                break

        return selected_orders

    def _evaluate_orders(
        self, selected_orders: list[int]
    ) -> tuple[list[int], list[int], int, float]:
        if not selected_orders:
            return [], [], 0, 0.0

        total_units = sum(self.order_units[order_idx] for order_idx in selected_orders)
        if total_units < self.lb or total_units > self.ub:
            return [], [], 0, 0.0

        demand = self._build_demand(selected_orders)
        visited_aisles = multi_greedy_aisle_select(dict(demand), self.aisles)

        if not visited_aisles:
            return [], [], 0, 0.0

        if not self._is_demand_coverable(demand, visited_aisles):
            return [], [], 0, 0.0

        objective = total_units / len(visited_aisles)

        return sorted(selected_orders), sorted(visited_aisles), total_units, objective

    def _is_better(
        self,
        units: int,
        aisles_count: int,
        objective: float,
        best_units: int,
        best_aisles_count: int,
        best_objective: float,
    ) -> bool:
        if objective > best_objective:
            return True
        if objective < best_objective:
            return False

        if units > best_units:
            return True
        if units < best_units:
            return False

        return aisles_count < best_aisles_count

    def _evaporate(self, pheromone: list[list[float]]) -> None:
        factor = 1.0 - self.evaporation
        for i in range(self.n_orders):
            for j in range(i + 1, self.n_orders):
                pheromone[i][j] = max(1e-9, pheromone[i][j] * factor)

    def _deposit(
        self,
        pheromone: list[list[float]],
        selected_orders: list[int],
        objective: float,
        weight: float = 1.0,
    ) -> None:
        if len(selected_orders) < 2:
            return

        deposit_value = max(1e-9, self.q_deposit * objective * weight)

        for i, j in combinations(sorted(selected_orders), 2):
            pheromone[i][j] += deposit_value

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        # Custom init logic
        self.n_ants = max(1, int(self.config.get("n_ants", 30)))
        self.n_iterations = max(1, int(self.config.get("n_iterations", 60)))

        self.alpha = float(self.config.get("alpha", 1.1))
        self.beta = float(self.config.get("beta", 2.2))

        self.evaporation = float(self.config.get("evaporation", 0.15))
        self.evaporation = min(max(self.evaporation, 0.0), 0.95)

        self.q_deposit = float(self.config.get("q_deposit", 1.0))
        self.initial_pheromone = float(self.config.get("initial_pheromone", 1.0))
        self.candidate_list_size = int(self.config.get("candidate_list_size", 0))

        self.stop_after_lb_prob = float(self.config.get("stop_after_lb_prob", 0.2))
        self.stop_after_lb_prob = min(max(self.stop_after_lb_prob, 0.0), 1.0)

        self.elitist_weight = float(self.config.get("elitist_weight", 2.0))

        self.random_seed = self.config.get("seed")

        self.order_units = [sum(order.values()) for order in self.orders]
        self.total_stock = self._build_total_stock()
        self.coverable_orders = self._compute_coverable_orders()

        item_aisle_count: dict[int, int] = {}
        for aisle in self.aisles:
            for item in aisle.keys():
                item_aisle_count[item] = item_aisle_count.get(item, 0) + 1

        self.item_rarity = {
            item: 1.0 / max(1, count) for item, count in item_aisle_count.items()
        }

        # Main solve logic
        if self.n_orders == 0 or self.n_aisles == 0:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        rng = random.Random(self.random_seed)

        pheromone = [
            [self.initial_pheromone for _ in range(self.n_orders)]
            for _ in range(self.n_orders)
        ]

        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_units = 0
        best_objective = 0.0

        for _ in range(self.n_iterations):
            iteration_solutions: list[tuple[list[int], list[int], int, float]] = []

            for _ in range(self.n_ants):
                selected_orders = self._construct_ant_solution(pheromone, rng)
                orders, aisles, units, objective = self._evaluate_orders(
                    selected_orders
                )

                if not orders:
                    continue

                iteration_solutions.append((orders, aisles, units, objective))

                if self._is_better(
                    units=units,
                    aisles_count=len(aisles),
                    objective=objective,
                    best_units=best_units,
                    best_aisles_count=len(best_aisles) if best_aisles else 10**9,
                    best_objective=best_objective,
                ):
                    best_orders = orders
                    best_aisles = aisles
                    best_units = units
                    best_objective = objective

            self._evaporate(pheromone)

            if not iteration_solutions:
                continue

            for orders, _, _, objective in iteration_solutions:
                self._deposit(pheromone, orders, objective)

            iter_best = max(
                iteration_solutions,
                key=lambda item: (item[3], item[2], -len(item[1])),
            )
            self._deposit(
                pheromone,
                iter_best[0],
                iter_best[3],
                weight=self.elitist_weight,
            )

        selected_orders = best_orders
        visited_aisles = best_aisles

        if not selected_orders or not visited_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}
