import math
import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.similarity import similarity
from problems.base import ProblemInput

_VALID_INIT = {"greedy", "grasp"}
_VALID_CONSTRUCTION = {"size", "synergy", "aisle_cost"}
_VALID_MOVES = {"swap", "drop", "add"}
_VALID_COOLING = {"geometric", "linear"}
_VALID_GREEDY = {"simple", "multi"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class SimulatedAnnealing(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        init = params.get("init")
        if init not in _VALID_INIT:
            raise ValueError(
                f"SimulatedAnnealing: invalid 'init'={init!r}; "
                f"expected one of {sorted(_VALID_INIT)}"
            )

        init_alpha = params.get("init_alpha", 0.3)
        if not isinstance(init_alpha, (int, float)) or isinstance(init_alpha, bool):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'init_alpha'={init_alpha!r}; "
                f"expected float in [0, 1]"
            )
        init_alpha = float(init_alpha)
        if not 0.0 <= init_alpha <= 1.0:
            raise ValueError(
                f"SimulatedAnnealing: 'init_alpha'={init_alpha} out of range [0, 1]"
            )

        init_construction = params.get("init_construction", "size")
        if init_construction not in _VALID_CONSTRUCTION:
            raise ValueError(
                f"SimulatedAnnealing: invalid 'init_construction'={init_construction!r}; "
                f"expected one of {sorted(_VALID_CONSTRUCTION)}"
            )

        moves = params.get("moves")
        if (
            not isinstance(moves, list)
            or len(moves) == 0
            or any(m not in _VALID_MOVES for m in moves)
        ):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'moves'={moves!r}; "
                f"expected non-empty list with values in {sorted(_VALID_MOVES)}"
            )

        cooling = params.get("cooling")
        if cooling not in _VALID_COOLING:
            raise ValueError(
                f"SimulatedAnnealing: invalid 'cooling'={cooling!r}; "
                f"expected one of {sorted(_VALID_COOLING)}"
            )

        initial_temp = params.get("initial_temp")
        if (
            not isinstance(initial_temp, (int, float))
            or isinstance(initial_temp, bool)
            or initial_temp <= 0
        ):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'initial_temp'={initial_temp!r}; "
                f"expected positive float"
            )

        cooling_rate = params.get("cooling_rate")
        if (
            not isinstance(cooling_rate, (int, float))
            or isinstance(cooling_rate, bool)
            or cooling_rate <= 0
        ):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'cooling_rate'={cooling_rate!r}; "
                f"expected positive float"
            )
        if cooling == "geometric" and cooling_rate >= 1.0:
            raise ValueError(
                f"SimulatedAnnealing: 'cooling_rate'={cooling_rate} must be in (0, 1) "
                f"for geometric cooling"
            )

        min_temp = params.get("min_temp", 1e-3)
        if (
            not isinstance(min_temp, (int, float))
            or isinstance(min_temp, bool)
            or min_temp <= 0
        ):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'min_temp'={min_temp!r}; "
                f"expected positive float"
            )

        max_iterations = params.get("max_iterations")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'max_iterations'={max_iterations!r}; "
                f"expected positive int"
            )

        greedy = params.get("greedy")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"SimulatedAnnealing: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )

        similarity_weighted = params.get("similarity_weighted", False)
        if not isinstance(similarity_weighted, bool):
            raise ValueError(
                f"SimulatedAnnealing: invalid 'similarity_weighted'={similarity_weighted!r}; "
                f"expected bool"
            )

        self._init = init
        self._init_alpha = init_alpha
        self._init_construction = init_construction
        self._moves = list(moves)
        self._cooling = cooling
        self._initial_temp = float(initial_temp)
        self._cooling_rate = float(cooling_rate)
        self._min_temp = float(min_temp)
        self._max_iterations = max_iterations
        self._greedy = greedy
        self._similarity_weighted = similarity_weighted
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "sa_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders
        n_aisles = instance.nAisles

        if n_orders == 0 or n_aisles == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in orders]
        stock_total = self._aggregate_stock(aisles)
        rng = random.Random(self._seed) if self._seed is not None else random.Random()

        current = self._build_initial(
            orders, aisles, order_sizes, stock_total, lb, ub, rng
        )
        if current is None:
            return dict(_EMPTY_RESULT)

        best = current
        temp = self._initial_temp

        for _ in range(self._max_iterations):
            if temp < self._min_temp:
                break

            move = rng.choice(self._moves)
            neighbor = self._generate_neighbor(
                current, move, orders, aisles, order_sizes, stock_total, lb, ub, rng
            )

            if neighbor is not None:
                delta = neighbor["objective"] - current["objective"]
                if delta > 0 or rng.random() < math.exp(delta / temp):
                    current = neighbor
                    if current["objective"] > best["objective"]:
                        best = current

            temp = self._cool(temp)

        return {
            "selected_orders": best["selected_orders"],
            "visited_aisles": best["visited_aisles"],
            "objective": best["objective"],
        }

    def _build_initial(
        self,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
        rng: random.Random,
    ) -> dict | None:
        if self._init == "grasp":
            return self._build_grasp(
                orders, aisles, order_sizes, stock_total, lb, ub, rng
            )
        return self._build_greedy(orders, aisles, order_sizes, stock_total, lb, ub)

    def _build_greedy(
        self,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
    ) -> dict | None:
        n_orders = len(orders)
        order_indices = sorted(
            (i for i in range(n_orders) if order_sizes[i] > 0),
            key=lambda i: order_sizes[i],
            reverse=True,
        )

        selected: list[int] = []
        demand: dict[int, int] = {}
        total_units = 0
        stock_remaining = dict(stock_total)

        for idx in order_indices:
            size = order_sizes[idx]
            if total_units + size > ub:
                continue
            order = orders[idx]
            if any(stock_remaining.get(it, 0) < q for it, q in order.items()):
                continue
            selected.append(idx)
            total_units += size
            for it, q in order.items():
                demand[it] = demand.get(it, 0) + q
                stock_remaining[it] -= q

        if total_units < lb:
            return None

        visited = self._select_aisles(demand, aisles)
        if not visited:
            return None

        return {
            "selected_orders": selected,
            "visited_aisles": visited,
            "objective": total_units / len(visited),
            "_demand": demand,
            "_total_units": total_units,
        }

    def _build_grasp(
        self,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
        rng: random.Random,
    ) -> dict | None:
        n_orders = len(orders)
        selected: list[int] = []
        demand: dict[int, int] = {}
        total_units = 0
        stock_remaining = dict(stock_total)
        remaining = set(range(n_orders))

        while remaining:
            feasible: list[int] = []
            for idx in remaining:
                if order_sizes[idx] == 0:
                    continue
                if total_units + order_sizes[idx] > ub:
                    continue
                order = orders[idx]
                ok = True
                for item, qty in order.items():
                    if stock_remaining.get(item, 0) < qty:
                        ok = False
                        break
                if ok:
                    feasible.append(idx)

            if not feasible:
                break

            scores = self._score_candidates(feasible, orders, aisles, demand, order_sizes)

            g_max = max(scores.values())
            g_min = min(scores.values())
            threshold = g_max - self._init_alpha * (g_max - g_min)
            rcl = [i for i, s in scores.items() if s >= threshold]

            pick = rng.choice(rcl)
            pick_order = orders[pick]

            selected.append(pick)
            total_units += order_sizes[pick]
            for item, qty in pick_order.items():
                demand[item] = demand.get(item, 0) + qty
                stock_remaining[item] -= qty
            remaining.remove(pick)

        if total_units < lb:
            return None

        visited = self._select_aisles(demand, aisles)
        if not visited:
            return None

        return {
            "selected_orders": list(selected),
            "visited_aisles": visited,
            "objective": total_units / len(visited),
            "_demand": demand,
            "_total_units": total_units,
        }

    def _score_candidates(
        self,
        feasible: list[int],
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        demand: dict[int, int],
        order_sizes: list[int],
    ) -> dict[int, float]:
        if self._init_construction == "size":
            return {i: float(order_sizes[i]) for i in feasible}

        if self._init_construction == "synergy":
            if not demand:
                return {i: float(order_sizes[i]) for i in feasible}
            return {
                i: similarity(
                    demand, orders[i], weighted=self._similarity_weighted
                )
                for i in feasible
            }

        if not demand:
            return {i: float(order_sizes[i]) for i in feasible}
        current_aisles = set(greedy_aisle_select(dict(demand), aisles))
        scores: dict[int, float] = {}
        for i in feasible:
            combined = dict(demand)
            for item, qty in orders[i].items():
                combined[item] = combined.get(item, 0) + qty
            after = set(greedy_aisle_select(combined, aisles))
            scores[i] = -float(len(after - current_aisles))
        return scores

    def _generate_neighbor(
        self,
        solution: dict,
        move: str,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
        rng: random.Random,
    ) -> dict | None:
        if move == "swap":
            return self._random_swap(
                solution, orders, aisles, order_sizes, stock_total, lb, ub, rng
            )
        if move == "drop":
            return self._random_drop(solution, orders, aisles, order_sizes, lb, rng)
        return self._random_add(
            solution, orders, aisles, order_sizes, stock_total, ub, rng
        )

    def _random_swap(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
        rng: random.Random,
    ) -> dict | None:
        selected = solution["selected_orders"]
        if not selected:
            return None
        selected_set = set(selected)
        demand = solution["_demand"]
        total_units = solution["_total_units"]
        n_orders = len(orders)

        unselected = [i for i in range(n_orders) if i not in selected_set and order_sizes[i] > 0]
        if not unselected:
            return None

        s_idx = rng.choice(selected)
        u_idx = rng.choice(unselected)

        s_order = orders[s_idx]
        s_size = order_sizes[s_idx]
        u_order = orders[u_idx]
        u_size = order_sizes[u_idx]

        new_total = total_units - s_size + u_size
        if new_total > ub or new_total < lb:
            return None

        new_demand = dict(demand)
        for item, qty in s_order.items():
            new_demand[item] = new_demand.get(item, 0) - qty
            if new_demand[item] == 0:
                del new_demand[item]
        for item, qty in u_order.items():
            new_demand[item] = new_demand.get(item, 0) + qty

        if not self._demand_within_stock(new_demand, stock_total):
            return None

        new_visited = self._select_aisles(new_demand, aisles)
        if not new_visited:
            return None

        new_selected = [i for i in selected if i != s_idx]
        new_selected.append(u_idx)
        return {
            "selected_orders": new_selected,
            "visited_aisles": new_visited,
            "objective": new_total / len(new_visited),
            "_demand": new_demand,
            "_total_units": new_total,
        }

    def _random_drop(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        lb: int,
        rng: random.Random,
    ) -> dict | None:
        selected = solution["selected_orders"]
        if len(selected) <= 1:
            return None

        demand = solution["_demand"]
        total_units = solution["_total_units"]

        s_idx = rng.choice(selected)
        s_order = orders[s_idx]
        s_size = order_sizes[s_idx]
        new_total = total_units - s_size
        if new_total < lb:
            return None

        new_demand = dict(demand)
        for item, qty in s_order.items():
            new_demand[item] = new_demand.get(item, 0) - qty
            if new_demand[item] == 0:
                del new_demand[item]

        new_visited = self._select_aisles(new_demand, aisles)
        if not new_visited:
            return None

        new_selected = [i for i in selected if i != s_idx]
        return {
            "selected_orders": new_selected,
            "visited_aisles": new_visited,
            "objective": new_total / len(new_visited),
            "_demand": new_demand,
            "_total_units": new_total,
        }

    def _random_add(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        ub: int,
        rng: random.Random,
    ) -> dict | None:
        selected = solution["selected_orders"]
        selected_set = set(selected)
        demand = solution["_demand"]
        total_units = solution["_total_units"]
        n_orders = len(orders)

        unselected = [i for i in range(n_orders) if i not in selected_set and order_sizes[i] > 0]
        if not unselected:
            return None

        u_idx = rng.choice(unselected)
        u_size = order_sizes[u_idx]
        new_total = total_units + u_size
        if new_total > ub:
            return None

        u_order = orders[u_idx]
        new_demand = dict(demand)
        for item, qty in u_order.items():
            new_demand[item] = new_demand.get(item, 0) + qty

        if not self._demand_within_stock(new_demand, stock_total):
            return None

        new_visited = self._select_aisles(new_demand, aisles)
        if not new_visited:
            return None

        new_selected = list(selected)
        new_selected.append(u_idx)
        return {
            "selected_orders": new_selected,
            "visited_aisles": new_visited,
            "objective": new_total / len(new_visited),
            "_demand": new_demand,
            "_total_units": new_total,
        }

    def _cool(self, temp: float) -> float:
        if self._cooling == "geometric":
            return temp * self._cooling_rate
        return max(self._min_temp, temp - self._cooling_rate)

    def _select_aisles(
        self, demand: dict[int, int], aisles: list[dict[int, int]]
    ) -> list[int]:
        if self._greedy == "multi":
            return multi_greedy_aisle_select(demand, aisles)
        return greedy_aisle_select(dict(demand), aisles)

    @staticmethod
    def _demand_within_stock(
        demand: dict[int, int], stock_total: dict[int, int]
    ) -> bool:
        for item, qty in demand.items():
            if stock_total.get(item, 0) < qty:
                return False
        return True

    @staticmethod
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
