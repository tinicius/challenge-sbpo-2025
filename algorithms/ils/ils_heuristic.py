import random
import time
from copy import deepcopy

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_OPERATORS = {"swap", "add", "remove"}
_VALID_STRATEGIES = {"first_improvement", "best_improvement"}


class ILSHeuristic(Algorithm):
    """Iterated Local Search for wave order-picking problem."""

    def __init__(self, params: dict):
        super().__init__(params)
        self._init_heuristic = params.get("initial_heuristic", "seed")
        self._ls_operators = params.get("ls_operators", ["swap", "add", "remove"])
        self._ls_strategy = params.get("ls_strategy", "first_improvement")
        self._ls_max_iter = int(params.get("ls_max_iterations", 100))
        self._neighbor_cap = int(params.get("neighbor_cap", 50))
        self._shake_k = int(params.get("shake_k", 3))
        self._shake_adaptive = bool(params.get("shake_adaptive", True))
        self._max_iter = int(params.get("max_iterations", 50))
        self._greedy = params.get("greedy", "simple")
        self._seed = params.get("seed")
        self._time_limit = params.get("time_limit")
        self.last_best = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        if self._greedy not in {"simple", "multi"}:
            raise ValueError(f"ILS: invalid greedy={self._greedy!r}")
        for op in self._ls_operators:
            if op not in _VALID_OPERATORS:
                raise ValueError(f"ILS: invalid operator={op!r}")
        if self._ls_strategy not in _VALID_STRATEGIES:
            raise ValueError(f"ILS: invalid strategy={self._ls_strategy!r}")

    @property
    def name(self) -> str:
        return "ils"

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
        rng = random.Random(self._seed)
        start_time = time.time()

        if instance.nOrders == 0 or instance.nAisles == 0:
            return dict(self.last_best)

        # Precompute order sizes
        order_sizes = [sum(o.values()) for o in instance.orders]
        n_orders = instance.nOrders
        lb, ub = instance.lb, instance.ub

        # Build initial solution
        current = self._build_initial_solution(instance, rng)
        if not current["selected_orders"]:
            return dict(self.last_best)

        # Local search on initial solution
        current = self._local_search(current, instance, order_sizes, rng, start_time)
        if current["objective"] > self.last_best["objective"]:
            self.last_best = dict(current)

        best = dict(current)
        shake_k = self._shake_k

        for iteration in range(self._max_iter):
            # Check time limit
            if self._time_limit and time.time() - start_time > self._time_limit:
                break

            # Perturbation
            perturbed = self._shake(dict(best), instance, order_sizes, shake_k, rng)

            # Local search
            improved = self._local_search(perturbed, instance, order_sizes, rng, start_time)

            # Acceptance: better-only
            if improved["objective"] > best["objective"]:
                best = dict(improved)
                if best["objective"] > self.last_best["objective"]:
                    self.last_best = dict(best)
                if self._shake_adaptive:
                    shake_k = max(1, self._shake_k)  # Reset perturbation strength
            elif self._shake_adaptive:
                shake_k = min(shake_k + 1, n_orders // 3)  # Increase perturbation

        return dict(self.last_best)

    def _build_initial_solution(self, instance: ProblemInput, rng: random.Random) -> dict:
        """Build initial solution using heuristic or greedy construction."""
        n_orders = instance.nOrders
        order_sizes = [sum(o.values()) for o in instance.orders]
        lb, ub = instance.lb, instance.ub

        # Try using existing heuristics
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

        # Fallback: greedy construction by order size
        return self._greedy_construction(instance, order_sizes, rng)

    def _greedy_construction(self, instance: ProblemInput, order_sizes: list, rng: random.Random) -> dict:
        """Greedy construction: add orders by descending size while feasible."""
        n_orders = instance.nOrders
        lb, ub = instance.lb, instance.ub

        ordered = sorted(range(n_orders), key=lambda i: order_sizes[i], reverse=True)

        selected = []
        total_units = 0
        demand = {}
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

    def _local_search(self, solution: dict, instance: ProblemInput, order_sizes: list, rng: random.Random, start_time: float = None) -> dict:
        """Apply local search operators to improve the solution."""
        # Evaluate input solution first (might not have objective set)
        if "objective" not in solution:
            self._evaluate(solution, instance)
        if solution.get("objective", 0.0) == 0.0:
            return solution

        best = dict(solution)
        best_obj = best["objective"]

        for _ in range(self._ls_max_iter):
            # Check time limit
            if start_time and self._time_limit and time.time() - start_time > self._time_limit:
                break

            improved = False

            for op in self._ls_operators:
                if op == "swap":
                    neighbors = self._swap_neighbors(best, instance, order_sizes, rng)
                elif op == "add":
                    neighbors = self._add_neighbors(best, instance, order_sizes, rng)
                else:  # remove
                    neighbors = self._remove_neighbors(best, instance, order_sizes, rng)

                for candidate in neighbors:
                    obj = self._evaluate(candidate, instance)
                    if obj > best_obj:
                        best = candidate
                        best_obj = obj
                        improved = True
                        if self._ls_strategy == "first_improvement":
                            break
                if improved and self._ls_strategy == "first_improvement":
                    break

            if not improved:
                break

        return best

    def _swap_neighbors(self, solution: dict, instance: ProblemInput, order_sizes: list, rng: random.Random) -> list:
        """Generate neighbors by swapping one selected order with one unselected."""
        selected = list(solution["selected_orders"])
        selected_set = set(selected)
        n_orders = instance.nOrders
        lb, ub = instance.lb, instance.ub

        # Get unselected orders
        unselected = [i for i in range(n_orders) if i not in selected_set]

        # Sample up to neighbor_cap swaps
        neighbors = []
        attempts = 0
        max_attempts = min(self._neighbor_cap * 10, len(selected) * len(unselected))

        while len(neighbors) < self._neighbor_cap and attempts < max_attempts:
            attempts += 1
            s = rng.choice(selected)
            u = rng.choice(unselected)
            new_selected = [o for o in selected if o != s] + [u]
            total = sum(order_sizes[i] for i in new_selected)
            if total < lb or total > ub:
                continue
            neighbors.append({"selected_orders": new_selected})

        return neighbors

    def _add_neighbors(self, solution: dict, instance: ProblemInput, order_sizes: list, rng: random.Random) -> list:
        """Generate neighbors by adding one unselected order."""
        selected = solution["selected_orders"]
        selected_set = set(selected)
        ub = instance.ub
        total_current = sum(order_sizes[i] for i in selected)

        # Get valid unselected orders
        unselected = [i for i in range(instance.nOrders) if i not in selected_set and total_current + order_sizes[i] <= ub]

        # Sample randomly
        neighbors = []
        if unselected:
            sample_size = min(self._neighbor_cap, len(unselected))
            sampled = rng.sample(unselected, sample_size)
            for u in sampled:
                neighbors.append({"selected_orders": selected + [u]})

        return neighbors

    def _remove_neighbors(self, solution: dict, instance: ProblemInput, order_sizes: list, rng: random.Random) -> list:
        """Generate neighbors by removing one selected order."""
        selected = solution["selected_orders"]
        lb = instance.lb

        if len(selected) <= 1:
            return []

        # Sample randomly up to neighbor_cap
        sample_size = min(self._neighbor_cap, len(selected))
        to_remove = rng.sample(selected, sample_size)

        neighbors = []
        for s in to_remove:
            new_selected = [o for o in selected if o != s]
            total = sum(order_sizes[i] for i in new_selected)
            if total < lb:
                continue
            neighbors.append({"selected_orders": new_selected})

        return neighbors

    def _shake(self, solution: dict, instance: ProblemInput, order_sizes: list, k: int, rng: random.Random) -> dict:
        """Perturb solution by randomly swapping k orders."""
        selected = list(solution["selected_orders"])
        n_orders = instance.nOrders
        lb, ub = instance.lb, instance.ub

        # Get unselected orders
        selected_set = set(selected)
        unselected = [i for i in range(n_orders) if i not in selected_set]

        if not selected or not unselected:
            return solution

        # Randomly select k orders to swap
        k_actual = min(k, len(selected), len(unselected))
        to_remove = rng.sample(selected, k_actual)
        to_add = rng.sample(unselected, k_actual)

        new_selected = [o for o in selected if o not in to_remove] + to_add
        total = sum(order_sizes[i] for i in new_selected)

        # Adjust if infeasible
        if total > ub:
            # Remove excess orders
            while total > ub and len(new_selected) > 0:
                idx = rng.randint(0, len(new_selected) - 1)
                total -= order_sizes[new_selected.pop(idx)]
        elif total < lb:
            # Try to add orders
            for u in unselected:
                if u in new_selected:
                    continue
                if total + order_sizes[u] <= ub:
                    new_selected.append(u)
                    total += order_sizes[u]
                    if total >= lb:
                        break

        return {"selected_orders": new_selected}

    def _evaluate(self, solution: dict, instance: ProblemInput) -> float:
        """Evaluate solution and update visited_aisles and objective."""
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

        demand = {}
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
        """Select aisles using configured greedy method."""
        if self._greedy == "multi":
            from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
            return multi_greedy_aisle_select(demand, aisles)
        return greedy_aisle_select(demand, aisles)

    @staticmethod
    def _aggregate_stock(aisles: list[dict]) -> dict:
        stock = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
