"""Tabu Search with short-term memory for wave order-picking.

Reference: Henn & Wäscher (EJOR 2012) — attribute-based tabu list,
aspiration criterion, best-improvement move selection.

Tabu attributes (indexed by order id):
  tabu_in[o]  — expiry iteration: o was removed; re-adding is prohibited.
  tabu_out[o] — expiry iteration: o was added;   re-removing is prohibited.

A move is tabu when any of its touched attributes are still active.
Aspiration overrides the tabu status when the resulting objective exceeds
the global best.
"""

import random
import time

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_OPERATORS = {"swap", "add", "remove"}


class TabuSearchHeuristic(Algorithm):
    """Tabu Search (short-term memory) for wave order-picking problem."""

    def __init__(self, params: dict):
        super().__init__(params)
        self._init_heuristic = params.get("initial_heuristic", "seed")
        self._ls_operators = params.get("ls_operators", ["swap", "add", "remove"])
        self._neighbor_cap = int(params.get("neighbor_cap", 50))
        self._tabu_tenure = int(params.get("tabu_tenure", 10))
        self._max_iter = int(params.get("max_iterations", 1000))
        self._max_idle_iter = int(params.get("max_idle_iterations", 200))
        self._aspiration = bool(params.get("aspiration", True))
        self._greedy = params.get("greedy", "simple")
        self._seed = params.get("seed")
        self._time_limit = params.get("time_limit")
        self.last_best = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        for op in self._ls_operators:
            if op not in _VALID_OPERATORS:
                raise ValueError(f"TabuSearch: invalid operator={op!r}")
        if self._greedy not in {"simple", "multi"}:
            raise ValueError(f"TabuSearch: invalid greedy={self._greedy!r}")

    @property
    def name(self) -> str:
        return "tabu"

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
        rng = random.Random(self._seed)
        start_time = time.time()

        if instance.nOrders == 0 or instance.nAisles == 0:
            return dict(self.last_best)

        order_sizes = [sum(o.values()) for o in instance.orders]
        n_orders = instance.nOrders

        current = self._build_initial_solution(instance, rng)
        if not current["selected_orders"]:
            return dict(self.last_best)

        self._evaluate(current, instance)
        if current["objective"] > 0.0:
            self.last_best = dict(current)

        global_best_obj = self.last_best["objective"]

        # tabu_in[o]  stores the expiry iteration (o was removed → re-add blocked)
        # tabu_out[o] stores the expiry iteration (o was added  → re-remove blocked)
        tabu_in = [0] * n_orders
        tabu_out = [0] * n_orders

        idle_iters = 0

        deadline = (start_time + self._time_limit) if self._time_limit else None

        for current_iter in range(self._max_iter):
            if deadline and time.time() >= deadline:
                break
            if idle_iters >= self._max_idle_iter:
                break

            neighbors = self._generate_neighbors(current, instance, order_sizes, rng)

            best_neighbor, best_obj, best_move = self._select_best_admissible(
                neighbors, tabu_in, tabu_out, current_iter, global_best_obj, instance,
                deadline,
            )

            if best_neighbor is None:
                idle_iters += 1
                continue

            current = best_neighbor
            self._apply_move(best_move, tabu_in, tabu_out, current_iter)

            if best_obj > global_best_obj:
                global_best_obj = best_obj
                self.last_best = dict(current)
                idle_iters = 0
            else:
                idle_iters += 1

        return dict(self.last_best)

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def _generate_neighbors(
        self,
        solution: dict,
        instance: ProblemInput,
        order_sizes: list,
        rng: random.Random,
    ) -> list:
        neighbors = []
        for op in self._ls_operators:
            if op == "swap":
                neighbors.extend(self._swap_neighbors(solution, instance, order_sizes, rng))
            elif op == "add":
                neighbors.extend(self._add_neighbors(solution, instance, order_sizes, rng))
            else:
                neighbors.extend(self._remove_neighbors(solution, instance, order_sizes, rng))
        return neighbors

    def _swap_neighbors(
        self,
        solution: dict,
        instance: ProblemInput,
        order_sizes: list,
        rng: random.Random,
    ) -> list:
        selected = list(solution["selected_orders"])
        selected_set = set(selected)
        lb, ub = instance.lb, instance.ub
        unselected = [i for i in range(instance.nOrders) if i not in selected_set]

        neighbors = []
        attempts = 0
        max_attempts = min(self._neighbor_cap * 10, max(1, len(selected) * len(unselected)))

        while len(neighbors) < self._neighbor_cap and attempts < max_attempts:
            attempts += 1
            s = rng.choice(selected)
            u = rng.choice(unselected)
            new_selected = [o for o in selected if o != s] + [u]
            total = sum(order_sizes[i] for i in new_selected)
            if total < lb or total > ub:
                continue
            neighbors.append({"selected_orders": new_selected, "move": ("swap", s, u)})

        return neighbors

    def _add_neighbors(
        self,
        solution: dict,
        instance: ProblemInput,
        order_sizes: list,
        rng: random.Random,
    ) -> list:
        selected = solution["selected_orders"]
        selected_set = set(selected)
        ub = instance.ub
        total_current = sum(order_sizes[i] for i in selected)

        unselected = [
            i for i in range(instance.nOrders)
            if i not in selected_set and total_current + order_sizes[i] <= ub
        ]

        neighbors = []
        if unselected:
            sample_size = min(self._neighbor_cap, len(unselected))
            for u in rng.sample(unselected, sample_size):
                neighbors.append({"selected_orders": selected + [u], "move": ("add", u)})

        return neighbors

    def _remove_neighbors(
        self,
        solution: dict,
        instance: ProblemInput,
        order_sizes: list,
        rng: random.Random,
    ) -> list:
        selected = solution["selected_orders"]
        lb = instance.lb

        if len(selected) <= 1:
            return []

        sample_size = min(self._neighbor_cap, len(selected))
        neighbors = []
        for s in rng.sample(selected, sample_size):
            new_selected = [o for o in selected if o != s]
            if sum(order_sizes[i] for i in new_selected) < lb:
                continue
            neighbors.append({"selected_orders": new_selected, "move": ("remove", s)})

        return neighbors

    # ------------------------------------------------------------------
    # Tabu management
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tabu(move: tuple, tabu_in: list, tabu_out: list, current_iter: int) -> bool:
        kind = move[0]
        if kind == "swap":
            _, s, u = move
            return tabu_out[s] > current_iter or tabu_in[u] > current_iter
        if kind == "add":
            return tabu_in[move[1]] > current_iter
        # remove
        return tabu_out[move[1]] > current_iter

    def _apply_move(
        self, move: tuple, tabu_in: list, tabu_out: list, current_iter: int
    ) -> None:
        expiry = current_iter + self._tabu_tenure
        kind = move[0]
        if kind == "swap":
            _, s, u = move
            tabu_in[s] = expiry   # s removed → re-adding blocked
            tabu_out[u] = expiry  # u added   → re-removing blocked
        elif kind == "add":
            tabu_out[move[1]] = expiry
        else:  # remove
            tabu_in[move[1]] = expiry

    def _select_best_admissible(
        self,
        neighbors: list,
        tabu_in: list,
        tabu_out: list,
        current_iter: int,
        global_best_obj: float,
        instance: ProblemInput,
        deadline: float | None = None,
    ) -> tuple:
        """Return (best_neighbor, best_obj, best_move) or (None, -1, None)."""
        best_neighbor = None
        best_obj = -1.0
        best_move = None

        # Fallback: best tabu move when no non-tabu+aspiration is found
        fallback_neighbor = None
        fallback_obj = -1.0
        fallback_move = None

        for candidate in neighbors:
            if deadline and time.time() >= deadline:
                break
            obj = self._evaluate(candidate, instance)
            if obj == 0.0:
                continue
            move = candidate["move"]
            is_tabu = self._is_tabu(move, tabu_in, tabu_out, current_iter)
            aspiration_ok = self._aspiration and obj > global_best_obj

            if not is_tabu or aspiration_ok:
                if obj > best_obj:
                    best_neighbor = candidate
                    best_obj = obj
                    best_move = move
            elif obj > fallback_obj:
                fallback_neighbor = candidate
                fallback_obj = obj
                fallback_move = move

        if best_neighbor is not None:
            return best_neighbor, best_obj, best_move

        # All moves tabu and none satisfies aspiration — use best tabu to avoid stall
        if fallback_neighbor is not None:
            return fallback_neighbor, fallback_obj, fallback_move

        return None, -1.0, None

    # ------------------------------------------------------------------
    # Evaluation and aisle selection (mirrors ILSHeuristic)
    # ------------------------------------------------------------------

    def _evaluate(self, solution: dict, instance: ProblemInput) -> float:
        selected = solution["selected_orders"]
        if not selected:
            solution["visited_aisles"] = []
            solution["objective"] = 0.0
            return 0.0

        total_units = sum(sum(instance.orders[i].values()) for i in selected)
        if total_units < instance.lb or total_units > instance.ub:
            solution["visited_aisles"] = []
            solution["objective"] = 0.0
            return 0.0

        demand: dict = {}
        for idx in selected:
            for item, qty in instance.orders[idx].items():
                demand[item] = demand.get(item, 0) + qty

        visited = self._select_aisles(demand, instance.aisles)
        if not visited:
            solution["visited_aisles"] = []
            solution["objective"] = 0.0
            return 0.0

        solution["visited_aisles"] = visited
        solution["objective"] = total_units / len(visited)
        return solution["objective"]

    def _select_aisles(self, demand: dict, aisles: list) -> list:
        if self._greedy == "multi":
            return multi_greedy_aisle_select(demand, aisles)
        return greedy_aisle_select(demand, aisles)

    # ------------------------------------------------------------------
    # Initial solution construction (mirrors ILSHeuristic)
    # ------------------------------------------------------------------

    def _build_initial_solution(self, instance: ProblemInput, rng: random.Random) -> dict:
        if self._init_heuristic == "seed":
            from algorithms.seed.seed_heuristic import SeedHeuristic
            try:
                result = SeedHeuristic({"greedy": self._greedy}).solve(instance)
                if result.get("selected_orders"):
                    return result
            except Exception:
                pass
        elif self._init_heuristic == "simple":
            from algorithms.simple.simple_heuristic import SimpleHeuristic
            try:
                result = SimpleHeuristic({"greedy": self._greedy}).solve(instance)
                if result.get("selected_orders"):
                    return result
            except Exception:
                pass
        elif self._init_heuristic == "aisle_first":
            from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
            try:
                result = AisleFirstHeuristic({"prune": self._greedy}).solve(instance)
                if result.get("selected_orders"):
                    return result
            except Exception:
                pass
        return self._greedy_construction(instance, rng)

    def _greedy_construction(self, instance: ProblemInput, rng: random.Random) -> dict:
        order_sizes = [sum(o.values()) for o in instance.orders]
        lb, ub = instance.lb, instance.ub
        ordered = sorted(range(instance.nOrders), key=lambda i: order_sizes[i], reverse=True)

        selected = []
        total_units = 0
        demand: dict = {}
        stock = self._aggregate_stock(instance.aisles)

        for idx in ordered:
            if total_units + order_sizes[idx] > ub:
                continue
            order = instance.orders[idx]
            if any(stock.get(it, 0) < q for it, q in order.items()):
                continue
            selected.append(idx)
            total_units += order_sizes[idx]
            for it, q in order.items():
                demand[it] = demand.get(it, 0) + q
                stock[it] = stock.get(it, 0) - q
            if total_units >= lb:
                break

        if total_units < lb:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        visited = self._select_aisles(demand, instance.aisles)
        if not visited:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        return {
            "selected_orders": selected,
            "visited_aisles": visited,
            "objective": total_units / len(visited),
        }

    @staticmethod
    def _aggregate_stock(aisles: list) -> dict:
        stock: dict = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
