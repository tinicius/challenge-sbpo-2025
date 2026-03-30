import random
import time
from collections import defaultdict

from ortools.linear_solver import pywraplp

from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class DinkelbachLocalSearchHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "dinkelbach_local_search"

    def _time_left(self, deadline: float) -> float:
        return max(0.0, deadline - time.perf_counter())

    def _total_units(self, orders: list[int]) -> int:
        return sum(self.units_per_order[o] for o in orders)

    def _build_demand(self, orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = defaultdict(int)
        for o in orders:
            for item, qty in self.orders[o].items():
                demand[item] += qty
        return demand

    def _build_inventory(self, aisles: set[int]) -> dict[int, int]:
        inventory: dict[int, int] = defaultdict(int)
        for aisle_idx in aisles:
            for item, qty in self.aisles[aisle_idx].items():
                inventory[item] += qty
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

    def _is_better_key(self, key, other_key, eps: float = 1e-9) -> bool:
        if key[0] > other_key[0] + eps:
            return True
        if abs(key[0] - other_key[0]) <= eps and key[1] > other_key[1]:
            return True
        if (
            abs(key[0] - other_key[0]) <= eps
            and key[1] == other_key[1]
            and key[2] > other_key[2]
        ):
            return True
        return False

    def _is_orders_feasible(self, orders: list[int]) -> bool:
        total = self._total_units(orders)
        if total < self.lb or total > self.ub:
            return False

        demand = self._build_demand(orders)
        for item, qty in demand.items():
            if qty > self.total_stock.get(item, 0):
                return False

        return True

    def _is_state_feasible(self, orders: list[int], aisles: list[int]) -> bool:
        if not orders or not aisles:
            return False
        if not self._is_orders_feasible(orders):
            return False

        demand = self._build_demand(orders)
        inventory = self._build_inventory(set(aisles))
        return not self._residual_demand(demand, inventory)

    def _repair_aisles_for_orders(
        self,
        orders: list[int],
        base_aisles: set[int] | None = None,
    ) -> list[int] | None:
        if not orders:
            return None

        demand = self._build_demand(orders)
        selected_aisles = set(base_aisles or set())

        if not selected_aisles:
            selected_aisles.update(multi_greedy_aisle_select(demand, self.aisles))

        while True:
            inventory = self._build_inventory(selected_aisles)
            residual = self._residual_demand(demand, inventory)
            if not residual:
                break

            best_aisle = None
            best_score = 0
            for aisle_idx in range(self.n_aisles):
                if aisle_idx in selected_aisles:
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

            selected_aisles.add(best_aisle)

        return self._cleanup_redundant_aisles(orders, sorted(selected_aisles))

    def _cleanup_redundant_aisles(
        self,
        orders: list[int],
        aisles: list[int],
    ) -> list[int]:
        if not orders or not aisles:
            return []

        demand = self._build_demand(orders)
        working = set(aisles)

        for aisle_idx in sorted(working):
            if len(working) <= 1:
                break

            candidate = set(working)
            candidate.remove(aisle_idx)
            inventory = self._build_inventory(candidate)
            if not self._residual_demand(demand, inventory):
                working = candidate

        return sorted(working)

    def _build_solution_from_orders(
        self,
        orders: list[int],
        base_aisles: set[int] | None = None,
    ) -> tuple[list[int], list[int]] | None:
        normalized = sorted(set(orders))
        if not self._is_orders_feasible(normalized):
            return None

        aisles = self._repair_aisles_for_orders(normalized, base_aisles=base_aisles)
        if aisles is None:
            return None

        if not self._is_state_feasible(normalized, aisles):
            return None

        return normalized, aisles

    def _parametric_key(
        self,
        orders: list[int],
        aisles: list[int],
        lambda_: float,
    ) -> tuple[float, float, float]:
        total = float(self._total_units(orders))
        aisle_count = float(len(aisles))
        score = total - lambda_ * aisle_count
        return (score, total, -aisle_count)

    def _construct_initial_orders(self) -> list[int] | None:
        ranked = list(range(self.n_orders))
        self.rng.shuffle(ranked)
        ranked.sort(key=lambda o: (-self.units_per_order[o], o))

        selected: list[int] = []
        demand: dict[int, int] = defaultdict(int)
        total = 0

        for o in ranked:
            units = self.units_per_order[o]
            if total + units > self.ub:
                continue

            feasible = True
            for item, qty in self.orders[o].items():
                if demand[item] + qty > self.total_stock.get(item, 0):
                    feasible = False
                    break

            if not feasible:
                continue

            selected.append(o)
            total += units
            for item, qty in self.orders[o].items():
                demand[item] += qty

            if total >= self.lb:
                break

        if total < self.lb:
            return None

        return sorted(selected)

    def _heuristic_parametric(
        self,
        lambda_: float,
        deadline: float,
        warm_orders: list[int] | None = None,
        warm_aisles: list[int] | None = None,
    ) -> tuple[list[int], list[int]] | None:
        if self._time_left(deadline) <= 0.01:
            return None

        starts: list[tuple[list[int], list[int], tuple[float, float, float]]] = []

        if warm_orders:
            warm_solution = self._build_solution_from_orders(
                warm_orders,
                base_aisles=set(warm_aisles or []),
            )
            if warm_solution is not None:
                orders, aisles = warm_solution
                starts.append(
                    (orders, aisles, self._parametric_key(orders, aisles, lambda_))
                )

        attempts = 0
        while attempts < self.local_restarts and self._time_left(deadline) > 0.01:
            initial_orders = self._construct_initial_orders()
            attempts += 1
            if initial_orders is None:
                continue
            solution = self._build_solution_from_orders(initial_orders)
            if solution is None:
                continue
            orders, aisles = solution
            starts.append(
                (orders, aisles, self._parametric_key(orders, aisles, lambda_))
            )

        if not starts:
            return None

        starts.sort(key=lambda x: x[2], reverse=True)
        current_orders, current_aisles, current_key = starts[0]
        best_orders, best_aisles, best_key = current_orders, current_aisles, current_key

        no_improve = 0

        while (
            self._time_left(deadline) > 0.01 and no_improve < self.local_max_no_improve
        ):
            improved = False

            order_set = set(current_orders)
            outside = [o for o in range(self.n_orders) if o not in order_set]
            self.rng.shuffle(outside)
            self.rng.shuffle(current_orders)

            outside = outside[: self.neighborhood_samples]
            in_wave = current_orders[: self.neighborhood_samples]

            for o_add in outside:
                if self._time_left(deadline) <= 0.01:
                    break
                candidate_orders = sorted(order_set | {o_add})
                candidate = self._build_solution_from_orders(
                    candidate_orders,
                    base_aisles=set(current_aisles),
                )
                if candidate is None:
                    continue
                cand_orders, cand_aisles = candidate
                cand_key = self._parametric_key(cand_orders, cand_aisles, lambda_)
                if self._is_better_key(cand_key, current_key):
                    current_orders, current_aisles, current_key = (
                        cand_orders,
                        cand_aisles,
                        cand_key,
                    )
                    improved = True
                    break

            if not improved:
                for o_rem in in_wave:
                    if self._time_left(deadline) <= 0.01:
                        break
                    candidate_orders = [o for o in order_set if o != o_rem]
                    candidate = self._build_solution_from_orders(
                        candidate_orders,
                        base_aisles=set(current_aisles),
                    )
                    if candidate is None:
                        continue
                    cand_orders, cand_aisles = candidate
                    cand_key = self._parametric_key(cand_orders, cand_aisles, lambda_)
                    if self._is_better_key(cand_key, current_key):
                        current_orders, current_aisles, current_key = (
                            cand_orders,
                            cand_aisles,
                            cand_key,
                        )
                        improved = True
                        break

            if not improved:
                for o_rem in in_wave:
                    if self._time_left(deadline) <= 0.01:
                        break
                    for o_add in outside:
                        if self._time_left(deadline) <= 0.01:
                            break
                        candidate_orders = sorted((order_set - {o_rem}) | {o_add})
                        candidate = self._build_solution_from_orders(
                            candidate_orders,
                            base_aisles=set(current_aisles),
                        )
                        if candidate is None:
                            continue
                        cand_orders, cand_aisles = candidate
                        cand_key = self._parametric_key(
                            cand_orders, cand_aisles, lambda_
                        )
                        if self._is_better_key(cand_key, current_key):
                            current_orders, current_aisles, current_key = (
                                cand_orders,
                                cand_aisles,
                                cand_key,
                            )
                            improved = True
                            break
                    if improved:
                        break

            if improved:
                if self._is_better_key(current_key, best_key):
                    best_orders, best_aisles, best_key = (
                        current_orders,
                        current_aisles,
                        current_key,
                    )
                no_improve = 0
            else:
                no_improve += 1

        return best_orders, best_aisles

    def _solve_parametric_milp(
        self,
        lambda_: float,
        time_limit_seconds: float,
    ) -> tuple[list[int], list[int]] | None:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            return None

        solver.SetTimeLimit(int(max(1.0, time_limit_seconds) * 1000))

        x = [solver.BoolVar(f"x_{o}") for o in range(self.n_orders)]
        y = [solver.BoolVar(f"y_{a}") for a in range(self.n_aisles)]

        objective = solver.Objective()
        for o in range(self.n_orders):
            objective.SetCoefficient(x[o], self.units_per_order[o])
        for a in range(self.n_aisles):
            objective.SetCoefficient(y[a], -lambda_)
        objective.SetMaximization()

        wave_ct = solver.Constraint(self.lb, self.ub, "wave_size")
        for o in range(self.n_orders):
            wave_ct.SetCoefficient(x[o], self.units_per_order[o])

        for item in self.relevant_items:
            ct = solver.Constraint(-solver.infinity(), 0, f"stock_{item}")
            for o, qty in self.items_in_orders[item]:
                ct.SetCoefficient(x[o], qty)
            for a, qty in self.items_in_aisles[item]:
                ct.SetCoefficient(y[a], -qty)

        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return None

        selected_orders = [
            o for o in range(self.n_orders) if x[o].solution_value() > 0.5
        ]
        selected_aisles = [
            a for a in range(self.n_aisles) if y[a].solution_value() > 0.5
        ]

        if not selected_orders or not selected_aisles:
            return None

        if not self._is_state_feasible(selected_orders, selected_aisles):
            repaired = self._build_solution_from_orders(
                selected_orders,
                base_aisles=set(selected_aisles),
            )
            return repaired

        compact_aisles = self._cleanup_redundant_aisles(
            selected_orders, selected_aisles
        )
        if self._is_state_feasible(selected_orders, compact_aisles):
            return sorted(selected_orders), compact_aisles

        return sorted(selected_orders), sorted(selected_aisles)

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
        self.rng = random.Random(self.config.get("random_seed"))
        self.max_iterations = max(1, int(self.config.get("max_iterations", 20)))
        self.epsilon = float(self.config.get("epsilon", 1e-6))
        self.time_limit_per_sub = float(self.config.get("time_limit", 2.0))
        self.total_time_limit = float(self.config.get("total_time_limit", 4.5))
        self.min_milp_time = float(self.config.get("min_milp_time", 0.15))
        self.refine_time = float(self.config.get("refine_time", 0.4))

        self.local_restarts = max(1, int(self.config.get("local_restarts", 3)))
        self.local_max_no_improve = max(
            1,
            int(self.config.get("local_max_no_improve", 4)),
        )
        self.neighborhood_samples = max(
            1,
            int(self.config.get("neighborhood_samples", 32)),
        )

        self.units_per_order = [
            sum(self.orders[o].values()) for o in range(self.n_orders)
        ]

        self.items_in_orders: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for o in range(self.n_orders):
            for item, qty in self.orders[o].items():
                self.items_in_orders[item].append((o, qty))

        self.items_in_aisles: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self.total_stock: dict[int, int] = defaultdict(int)
        for a in range(self.n_aisles):
            for item, qty in self.aisles[a].items():
                self.items_in_aisles[item].append((a, qty))
                self.total_stock[item] += qty

        self.relevant_items = set(self.items_in_orders.keys())

        # Main solve logic
        start = time.perf_counter()
        deadline = start + self.total_time_limit

        lambda_ = 0.0
        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_key = (-float("inf"), -float("inf"), -float("inf"))

        warm_orders: list[int] | None = None
        warm_aisles: list[int] | None = None

        for _ in range(self.max_iterations):
            if self._time_left(deadline) <= 0.02:
                break

            candidate = None
            sub_time = min(
                self.time_limit_per_sub, max(0.0, self._time_left(deadline) - 0.01)
            )
            if sub_time >= self.min_milp_time:
                candidate = self._solve_parametric_milp(lambda_, sub_time)

            if candidate is not None and self._time_left(deadline) > 0.05:
                local_deadline = min(
                    deadline,
                    time.perf_counter()
                    + min(self.refine_time, max(0.0, self._time_left(deadline) - 0.01)),
                )
                candidate = self._heuristic_parametric(
                    lambda_,
                    local_deadline,
                    warm_orders=candidate[0],
                    warm_aisles=candidate[1],
                )

            if candidate is None:
                fallback_deadline = min(
                    deadline,
                    time.perf_counter() + max(0.2, 0.75 * self._time_left(deadline)),
                )
                candidate = self._heuristic_parametric(
                    lambda_,
                    fallback_deadline,
                    warm_orders=warm_orders,
                    warm_aisles=warm_aisles,
                )

            if candidate is None:
                break

            selected_orders, selected_aisles = candidate
            total_units = self._total_units(selected_orders)
            aisle_count = len(selected_aisles)
            if aisle_count == 0:
                break

            ratio = total_units / aisle_count
            key = (ratio, float(total_units), -float(aisle_count))
            if self._is_better_key(key, best_key):
                best_key = key
                best_orders = selected_orders
                best_aisles = selected_aisles

            warm_orders = selected_orders
            warm_aisles = selected_aisles

            residual = total_units - lambda_ * aisle_count
            if abs(residual) <= self.epsilon:
                break

            lambda_ = ratio

        selected_orders = best_orders
        visited_aisles = best_aisles

        if not selected_orders or not visited_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}
