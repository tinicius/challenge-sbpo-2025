import random

from algorithms.base import Algorithm
from algorithms.grasp.local_search import improve_orders
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.similarity import similarity
from problems.base import ProblemInput

_VALID_CONSTRUCTION = {"size", "synergy", "aisle_cost", "aisle_cost_fast"}
_VALID_GREEDY = {"simple", "multi"}
_VALID_LOCAL_SEARCH = {"none", "swap", "full"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class GraspHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        alpha = params.get("alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError(
                f"GraspHeuristic: invalid 'alpha'={alpha!r}; expected float in [0, 1]"
            )
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"GraspHeuristic: 'alpha'={alpha} out of range; expected [0, 1]"
            )

        construction = params.get("construction_score")
        if construction not in _VALID_CONSTRUCTION:
            raise ValueError(
                f"GraspHeuristic: invalid 'construction_score'={construction!r}; "
                f"expected one of {sorted(_VALID_CONSTRUCTION)}"
            )

        max_iterations = params.get("max_iterations")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError(
                f"GraspHeuristic: invalid 'max_iterations'={max_iterations!r}; "
                f"expected positive int"
            )

        greedy = params.get("greedy")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"GraspHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )

        local_search = params.get("local_search")
        if local_search not in _VALID_LOCAL_SEARCH:
            raise ValueError(
                f"GraspHeuristic: invalid 'local_search'={local_search!r}; "
                f"expected one of {sorted(_VALID_LOCAL_SEARCH)}"
            )

        similarity_weighted = params.get("similarity_weighted", False)
        if not isinstance(similarity_weighted, bool):
            raise ValueError(
                f"GraspHeuristic: invalid 'similarity_weighted'={similarity_weighted!r}; "
                f"expected bool"
            )

        self._alpha = alpha
        self._construction = construction
        self._max_iterations = max_iterations
        self._greedy = greedy
        self._local_search = local_search
        self._similarity_weighted = similarity_weighted
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "grasp_heuristic"

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

        best_aisle_for_item = (
            self._compute_best_aisle_for_item(aisles)
            if self._construction == "aisle_cost_fast"
            else None
        )

        self.last_best = dict(_EMPTY_RESULT)

        for _ in range(self._max_iterations):
            built = self._construct(
                orders, aisles, order_sizes, stock_total, lb, ub, rng,
                best_aisle_for_item,
            )
            if built is None:
                continue

            if built["objective"] > self.last_best["objective"]:
                self.last_best = built

            if self._local_search != "none":
                built = self._local_search_improve(
                    built, orders, aisles, order_sizes, stock_total, lb, ub
                )
                if built["objective"] > self.last_best["objective"]:
                    self.last_best = built

        return {
            "selected_orders": self.last_best["selected_orders"],
            "visited_aisles": self.last_best["visited_aisles"],
            "objective": self.last_best["objective"],
        }

    def _construct(
        self,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
        rng: random.Random,
        best_aisle_for_item: dict[int, int] | None = None,
    ) -> dict | None:
        n_orders = len(orders)
        selected: list[int] = []
        demand: dict[int, int] = {}
        total_units = 0
        stock_remaining = dict(stock_total)
        remaining = set(range(n_orders))
        current_aisles_set: set[int] = set()

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

            scores = self._score_candidates(
                feasible, orders, aisles, demand, order_sizes,
                current_aisles_set, best_aisle_for_item,
            )

            g_max = max(scores.values())
            g_min = min(scores.values())
            threshold = g_max - self._alpha * (g_max - g_min)
            rcl = [i for i, s in scores.items() if s >= threshold]

            pick = rng.choice(rcl)
            pick_order = orders[pick]

            selected.append(pick)
            total_units += order_sizes[pick]
            for item, qty in pick_order.items():
                demand[item] = demand.get(item, 0) + qty
                stock_remaining[item] -= qty
                if best_aisle_for_item is not None:
                    a = best_aisle_for_item.get(item)
                    if a is not None:
                        current_aisles_set.add(a)
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
        current_aisles_set: set[int] | None = None,
        best_aisle_for_item: dict[int, int] | None = None,
    ) -> dict[int, float]:
        if self._construction == "size":
            return {i: float(order_sizes[i]) for i in feasible}

        if self._construction == "synergy":
            if not demand:
                return {i: float(order_sizes[i]) for i in feasible}
            return {
                i: similarity(
                    demand, orders[i], weighted=self._similarity_weighted
                )
                for i in feasible
            }

        if self._construction == "aisle_cost_fast":
            if not demand:
                return {i: float(order_sizes[i]) for i in feasible}
            scores: dict[int, float] = {}
            for i in feasible:
                shortfall = 0
                for item, qty in orders[i].items():
                    a = best_aisle_for_item.get(item) if best_aisle_for_item else None
                    if a is None or a not in current_aisles_set:
                        shortfall += qty
                scores[i] = -float(shortfall)
            return scores

        # "aisle_cost": higher score = fewer new aisles added
        if not demand:
            return {i: float(order_sizes[i]) for i in feasible}
        current_aisles = set(greedy_aisle_select(dict(demand), aisles))
        scores = {}
        for i in feasible:
            combined = dict(demand)
            for item, qty in orders[i].items():
                combined[item] = combined.get(item, 0) + qty
            after = set(greedy_aisle_select(combined, aisles))
            scores[i] = -float(len(after - current_aisles))
        return scores

    def _select_aisles(
        self, demand: dict[int, int], aisles: list[dict[int, int]]
    ) -> list[int]:
        if self._greedy == "multi":
            return multi_greedy_aisle_select(demand, aisles)
        return greedy_aisle_select(dict(demand), aisles)

    def _local_search_improve(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        stock_total: dict[int, int],
        lb: int,
        ub: int,
    ) -> dict:
        return improve_orders(
            solution, orders, aisles, order_sizes, stock_total, lb, ub,
            self._select_aisles, self._local_search,
        )

    @staticmethod
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock

    @staticmethod
    def _compute_best_aisle_for_item(
        aisles: list[dict[int, int]],
    ) -> dict[int, int]:
        best: dict[int, int] = {}
        best_qty: dict[int, int] = {}
        for a_idx, aisle in enumerate(aisles):
            for item, qty in aisle.items():
                if qty > best_qty.get(item, -1):
                    best[item] = a_idx
                    best_qty[item] = qty
        return best
