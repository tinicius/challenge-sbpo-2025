import random

from algorithms.base import Algorithm
from algorithms.grasp.local_search import improve_orders
from algorithms.utils.aisle_rank import (
    VALID_AISLE_SCORE,
    VALID_ORDER_MODE,
    aggregate_demand,
    aggregate_demand_from,
    build_order_sequence,
    pack_orders,
    score_aisles,
)
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_SCORING = {"static", "adaptive"}
_VALID_GREEDY = {"simple", "multi"}
_VALID_LOCAL_SEARCH = {"none", "swap", "full"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class AisleGraspHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        alpha = params.get("alpha")
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'alpha'={alpha!r}; expected float in [0, 1]"
            )
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"AisleGraspHeuristic: 'alpha'={alpha} out of range; expected [0, 1]"
            )

        max_iterations = params.get("max_iterations")
        if (
            not isinstance(max_iterations, int)
            or isinstance(max_iterations, bool)
            or max_iterations <= 0
        ):
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'max_iterations'={max_iterations!r}; "
                f"expected positive int"
            )

        scoring = params.get("scoring", "static")
        if scoring not in _VALID_SCORING:
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'scoring'={scoring!r}; "
                f"expected one of {sorted(_VALID_SCORING)}"
            )

        aisle_score = params.get("aisle_score", "useful")
        if aisle_score not in VALID_AISLE_SCORE:
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'aisle_score'={aisle_score!r}; "
                f"expected one of {sorted(VALID_AISLE_SCORE)}"
            )

        packing_order = params.get("packing_order")
        if packing_order not in VALID_ORDER_MODE:
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'packing_order'={packing_order!r}; "
                f"expected one of {sorted(v for v in VALID_ORDER_MODE if v)} or unset"
            )

        greedy = params.get("greedy", "simple")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )

        local_search_aisle = params.get("local_search_aisle", "none")
        if local_search_aisle not in _VALID_LOCAL_SEARCH:
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'local_search_aisle'={local_search_aisle!r}; "
                f"expected one of {sorted(_VALID_LOCAL_SEARCH)}"
            )

        local_search_order = params.get("local_search_order", "none")
        if local_search_order not in _VALID_LOCAL_SEARCH:
            raise ValueError(
                f"AisleGraspHeuristic: invalid 'local_search_order'={local_search_order!r}; "
                f"expected one of {sorted(_VALID_LOCAL_SEARCH)}"
            )

        self._alpha = alpha
        self._max_iterations = max_iterations
        self._scoring = scoring
        self._aisle_score = aisle_score
        self._packing_order = packing_order
        self._greedy = greedy
        self._local_search_aisle = local_search_aisle
        self._local_search_order = local_search_order
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "aisle_grasp_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders
        n_aisles = instance.nAisles

        if n_orders == 0 or n_aisles == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in orders]
        total_demand = aggregate_demand(orders)
        stock_total = self._aggregate_stock(aisles)
        rng = random.Random(self._seed) if self._seed is not None else random.Random()

        # Pre-compute static scores once per solve.
        static_scores = score_aisles(
            aisles, range(n_aisles), self._aisle_score, total_demand
        )

        best = dict(_EMPTY_RESULT)

        for _ in range(self._max_iterations):
            built = self._construct(
                orders, aisles, order_sizes, total_demand,
                static_scores, lb, ub, rng,
            )
            if built is None:
                continue

            if built["objective"] > best["objective"]:
                best = {k: v for k, v in built.items() if not k.startswith("_")}

            if self._local_search_aisle != "none":
                built = self._improve_aisles(
                    built, orders, aisles, order_sizes, lb, ub,
                )

            if self._local_search_order != "none":
                built = improve_orders(
                    built, orders, aisles, order_sizes, stock_total, lb, ub,
                    self._select_aisles, self._local_search_order,
                )

            if built["objective"] > best["objective"]:
                best = {k: v for k, v in built.items() if not k.startswith("_")}

        return best

    def _construct(
        self,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        total_demand: dict[int, int],
        static_scores: dict[int, float],
        lb: int,
        ub: int,
        rng: random.Random,
    ) -> dict | None:
        n_orders = len(orders)
        n_aisles = len(aisles)

        packing_seq = build_order_sequence(
            n_orders, order_sizes, self._packing_order, rng=rng
        )

        inventory: dict[int, int] = {}
        remaining = set(range(n_aisles))
        ordered: list[int] = []

        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_units = 0
        best_obj = 0.0

        while remaining:
            if self._scoring == "static":
                scores = {a: static_scores[a] for a in remaining}
            else:
                gap = {
                    item: max(0, qty - inventory.get(item, 0))
                    for item, qty in total_demand.items()
                }
                scores = score_aisles(aisles, remaining, self._aisle_score, gap)

            g_max = max(scores.values())
            g_min = min(scores.values())
            if g_max <= 0:
                # No remaining aisle contributes positively; stop expansion.
                break

            threshold = g_max - self._alpha * (g_max - g_min)
            rcl = [a for a, s in scores.items() if s >= threshold]
            pick = rng.choice(rcl)

            ordered.append(pick)
            remaining.discard(pick)
            for item, qty in aisles[pick].items():
                inventory[item] = inventory.get(item, 0) + qty

            k = len(ordered)
            if ub / k <= best_obj:
                break

            selected, total = pack_orders(
                packing_seq, orders, order_sizes, inventory, ub
            )
            if total < lb:
                continue
            obj = total / k
            if obj > best_obj:
                best_obj = obj
                best_units = total
                best_orders = selected
                best_aisles = list(ordered)

        if not best_orders:
            return None

        return {
            "selected_orders": best_orders,
            "visited_aisles": best_aisles,
            "objective": best_obj,
            "_demand": aggregate_demand_from(orders, best_orders),
            "_total_units": best_units,
            "_packing_seq": packing_seq,
        }

    def _improve_aisles(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        lb: int,
        ub: int,
    ) -> dict:
        current = solution
        improved = True
        while improved:
            improved = False

            swap_move = self._try_aisle_swap(
                current, orders, aisles, order_sizes, lb, ub
            )
            if swap_move is not None:
                current = swap_move
                improved = True
                continue

            if self._local_search_aisle == "full":
                drop_move = self._try_aisle_drop(
                    current, orders, aisles, order_sizes, lb, ub
                )
                if drop_move is not None:
                    current = drop_move
                    improved = True
                    continue

                add_move = self._try_aisle_add(
                    current, orders, aisles, order_sizes, lb, ub
                )
                if add_move is not None:
                    current = add_move
                    improved = True
                    continue

        return current

    def _try_aisle_swap(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        lb: int,
        ub: int,
    ) -> dict | None:
        visited = solution["visited_aisles"]
        packing_seq = solution["_packing_seq"]
        base_obj = solution["objective"]
        visited_set = set(visited)
        k = len(visited)
        n_aisles = len(aisles)

        base_inventory = self._build_inventory(aisles, visited)

        for v in visited:
            inv_minus = self._subtract_aisle(base_inventory, aisles[v])
            for u in range(n_aisles):
                if u in visited_set:
                    continue
                inv_new = self._add_aisle(inv_minus, aisles[u])
                selected, total = pack_orders(
                    packing_seq, orders, order_sizes, inv_new, ub
                )
                if total < lb:
                    continue
                new_obj = total / k
                if new_obj > base_obj:
                    new_visited = [a for a in visited if a != v]
                    new_visited.append(u)
                    return self._make_solution(
                        selected, new_visited, total, orders, packing_seq
                    )
        return None

    def _try_aisle_drop(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        lb: int,
        ub: int,
    ) -> dict | None:
        visited = solution["visited_aisles"]
        packing_seq = solution["_packing_seq"]
        base_obj = solution["objective"]
        k = len(visited)
        if k <= 1:
            return None

        base_inventory = self._build_inventory(aisles, visited)

        for v in visited:
            inv_new = self._subtract_aisle(base_inventory, aisles[v])
            selected, total = pack_orders(
                packing_seq, orders, order_sizes, inv_new, ub
            )
            if total < lb:
                continue
            new_k = k - 1
            new_obj = total / new_k
            if new_obj > base_obj:
                new_visited = [a for a in visited if a != v]
                return self._make_solution(
                    selected, new_visited, total, orders, packing_seq
                )
        return None

    def _try_aisle_add(
        self,
        solution: dict,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
        lb: int,
        ub: int,
    ) -> dict | None:
        visited = solution["visited_aisles"]
        packing_seq = solution["_packing_seq"]
        base_obj = solution["objective"]
        visited_set = set(visited)
        k = len(visited)
        n_aisles = len(aisles)

        base_inventory = self._build_inventory(aisles, visited)

        for u in range(n_aisles):
            if u in visited_set:
                continue
            inv_new = self._add_aisle(base_inventory, aisles[u])
            selected, total = pack_orders(
                packing_seq, orders, order_sizes, inv_new, ub
            )
            if total < lb:
                continue
            new_k = k + 1
            new_obj = total / new_k
            if new_obj > base_obj:
                new_visited = list(visited)
                new_visited.append(u)
                return self._make_solution(
                    selected, new_visited, total, orders, packing_seq
                )
        return None

    def _select_aisles(
        self, demand: dict[int, int], aisles: list[dict[int, int]]
    ) -> list[int]:
        if self._greedy == "multi":
            return multi_greedy_aisle_select(demand, aisles)
        return greedy_aisle_select(dict(demand), aisles)

    @staticmethod
    def _build_inventory(
        aisles: list[dict[int, int]], visited: list[int]
    ) -> dict[int, int]:
        inv: dict[int, int] = {}
        for a in visited:
            for item, qty in aisles[a].items():
                inv[item] = inv.get(item, 0) + qty
        return inv

    @staticmethod
    def _add_aisle(
        inventory: dict[int, int], aisle: dict[int, int]
    ) -> dict[int, int]:
        new = dict(inventory)
        for item, qty in aisle.items():
            new[item] = new.get(item, 0) + qty
        return new

    @staticmethod
    def _subtract_aisle(
        inventory: dict[int, int], aisle: dict[int, int]
    ) -> dict[int, int]:
        new = dict(inventory)
        for item, qty in aisle.items():
            left = new.get(item, 0) - qty
            if left > 0:
                new[item] = left
            elif item in new:
                del new[item]
        return new

    @staticmethod
    def _make_solution(
        selected_orders: list[int],
        visited_aisles: list[int],
        total_units: int,
        orders: list[dict[int, int]],
        packing_seq: list[int],
    ) -> dict:
        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": total_units / len(visited_aisles),
            "_demand": aggregate_demand_from(orders, selected_orders),
            "_total_units": total_units,
            "_packing_seq": packing_seq,
        }

    @staticmethod
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
