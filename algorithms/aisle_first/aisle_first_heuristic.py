from algorithms.aisle_first._local_search import (
    apply_local_search,
    validate_local_search,
)
from algorithms.base import Algorithm
from algorithms.utils.aisle_rank import (
    VALID_AISLE_SCORE,
    VALID_ORDER_MODE,
    aggregate_demand,
    aggregate_demand_from,
    build_order_sequence,
    pack_orders,
    rank_aisles,
)
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_PRUNE = {None, "simple", "multi"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class AisleFirstHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)
        score = params.get("score", "useful")
        if score not in VALID_AISLE_SCORE:
            raise ValueError(
                f"AisleFirstHeuristic: invalid 'score'={score!r}; "
                f"expected one of {sorted(VALID_AISLE_SCORE)}"
            )
        order = params.get("order")
        if order not in VALID_ORDER_MODE:
            raise ValueError(
                f"AisleFirstHeuristic: invalid 'order'={order!r}; "
                f"expected one of {sorted(v for v in VALID_ORDER_MODE if v)} or unset"
            )
        prune = params.get("prune")
        if prune not in _VALID_PRUNE:
            raise ValueError(
                f"AisleFirstHeuristic: invalid 'prune'={prune!r}; "
                f"expected one of {sorted(v for v in _VALID_PRUNE if v)} or unset"
            )
        self._score = score
        self._order = order
        self._prune = prune
        self._seed = params.get("seed")
        self._local_search = validate_local_search(
            params.get("local_search"), owner="AisleFirstHeuristic"
        )

    @property
    def name(self) -> str:
        return "aisle_first_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders
        n_aisles = instance.nAisles

        if n_aisles == 0 or n_orders == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in orders]
        total_demand = aggregate_demand(orders)

        ranked_aisles = rank_aisles(aisles, self._score, total_demand)
        order_sequence = build_order_sequence(
            n_orders, order_sizes, self._order, seed=self._seed
        )

        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_obj = 0.0

        inventory: dict[int, int] = {}
        for k, aisle_idx in enumerate(ranked_aisles, start=1):
            for item, qty in aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty

            # Early-stop only valid when prune is disabled: with prune the
            # achievable obj for this k is total_units / |visited| ≥ ub / k,
            # so the bound ub/k ≤ best_obj is no longer a safe cutoff.
            if self._prune is None and ub / k <= best_obj:
                break

            selected_orders, total_units = pack_orders(
                order_sequence, orders, order_sizes, inventory, ub
            )
            if total_units < lb:
                continue

            if self._prune is not None:
                demand = aggregate_demand_from(orders, selected_orders)
                visited = (
                    multi_greedy_aisle_select(demand, aisles)
                    if self._prune == "multi"
                    else greedy_aisle_select(demand, aisles)
                )
                if not visited:
                    continue
                obj = total_units / len(visited)
                aisles_for_k = visited
            else:
                obj = total_units / k
                aisles_for_k = ranked_aisles[:k]

            if obj > best_obj:
                best_obj = obj
                best_orders = selected_orders
                best_aisles = aisles_for_k

        if not best_orders:
            return dict(_EMPTY_RESULT)

        result = {
            "selected_orders": best_orders,
            "visited_aisles": best_aisles,
            "objective": best_obj,
        }

        if self._local_search is not None:
            result = apply_local_search(
                result,
                instance,
                self._prune,
                self._local_search,
                order_sequence,
                order_sizes,
            )

        return result
