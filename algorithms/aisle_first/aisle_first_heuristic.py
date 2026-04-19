import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_SCORE = {"useful", "units", "variety", "mixed"}
_VALID_ORDER = {None, "asc", "desc"}
_VALID_PRUNE = {None, "simple", "multi"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class AisleFirstHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)
        score = params.get("score", "useful")
        if score not in _VALID_SCORE:
            raise ValueError(
                f"AisleFirstHeuristic: invalid 'score'={score!r}; "
                f"expected one of {sorted(_VALID_SCORE)}"
            )
        order = params.get("order")
        if order not in _VALID_ORDER:
            raise ValueError(
                f"AisleFirstHeuristic: invalid 'order'={order!r}; "
                f"expected one of {sorted(v for v in _VALID_ORDER if v)} or unset"
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
        total_demand = self._aggregate_demand(orders)

        ranked_aisles = self._rank_aisles(aisles, total_demand)
        order_sequence = self._build_order_sequence(n_orders, order_sizes)

        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_units = 0
        best_obj = 0.0

        inventory: dict[int, int] = {}
        for k, aisle_idx in enumerate(ranked_aisles, start=1):
            for item, qty in aisles[aisle_idx].items():
                inventory[item] = inventory.get(item, 0) + qty

            if ub / k <= best_obj:
                break

            selected_orders, total_units = self._pack_orders(
                order_sequence, orders, order_sizes, inventory, ub
            )
            if total_units < lb:
                continue

            obj = total_units / k
            if obj > best_obj:
                best_obj = obj
                best_units = total_units
                best_orders = selected_orders
                best_aisles = ranked_aisles[:k]

        if not best_orders:
            return dict(_EMPTY_RESULT)

        if self._prune is not None:
            demand = self._aggregate_demand_from(orders, best_orders)
            visited = (
                multi_greedy_aisle_select(demand, aisles)
                if self._prune == "multi"
                else greedy_aisle_select(demand, aisles)
            )
            if not visited:
                return dict(_EMPTY_RESULT)
            return {
                "selected_orders": best_orders,
                "visited_aisles": visited,
                "objective": best_units / len(visited),
            }

        return {
            "selected_orders": best_orders,
            "visited_aisles": best_aisles,
            "objective": best_obj,
        }

    def _rank_aisles(
        self,
        aisles: list[dict[int, int]],
        total_demand: dict[int, int],
    ) -> list[int]:
        if self._score == "useful":
            def score(a: dict[int, int]) -> int:
                return sum(
                    min(qty, total_demand.get(item, 0)) for item, qty in a.items()
                )
        elif self._score == "units":
            def score(a: dict[int, int]) -> int:
                return sum(a.values())
        elif self._score == "variety":
            def score(a: dict[int, int]) -> int:
                return len(a)
        else:  # "mixed"
            def score(a: dict[int, int]) -> int:
                return sum(a.values()) * len(a)

        return sorted(
            range(len(aisles)),
            key=lambda idx: score(aisles[idx]),
            reverse=True,
        )

    def _build_order_sequence(
        self, n_orders: int, order_sizes: list[int]
    ) -> list[int]:
        if self._order is None:
            rng = random.Random(self._seed) if self._seed is not None else random
            indices = list(range(n_orders))
            rng.shuffle(indices)
            return indices
        return sorted(
            range(n_orders),
            key=order_sizes.__getitem__,
            reverse=(self._order == "desc"),
        )

    @staticmethod
    def _aggregate_demand(orders: list[dict[int, int]]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for o in orders:
            for item, qty in o.items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    @staticmethod
    def _aggregate_demand_from(
        orders: list[dict[int, int]], selected: list[int]
    ) -> dict[int, int]:
        demand: dict[int, int] = {}
        for idx in selected:
            for item, qty in orders[idx].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    @staticmethod
    def _pack_orders(
        sequence: list[int],
        orders: list[dict[int, int]],
        order_sizes: list[int],
        inventory: dict[int, int],
        ub: int,
    ) -> tuple[list[int], int]:
        remaining = dict(inventory)
        selected: list[int] = []
        total = 0
        for idx in sequence:
            size = order_sizes[idx]
            if total + size > ub:
                continue
            order = orders[idx]
            if any(remaining.get(item, 0) < qty for item, qty in order.items()):
                continue
            selected.append(idx)
            total += size
            for item, qty in order.items():
                remaining[item] -= qty
        return selected, total
