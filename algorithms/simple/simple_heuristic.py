import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_ORDER = {None, "asc", "desc"}
_VALID_GREEDY = {"simple", "multi"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class SimpleHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)
        order = params.get("order")
        if order not in _VALID_ORDER:
            raise ValueError(
                f"SimpleHeuristic: invalid 'order'={order!r}; "
                f"expected one of {sorted(v for v in _VALID_ORDER if v)} or unset"
            )
        greedy = params.get("greedy")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"SimpleHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )
        self._order = order
        self._greedy = greedy
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "simple_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub

        order_sizes = [sum(o.values()) for o in orders]
        indices = self._build_traversal(instance.nOrders, order_sizes)

        stock = self._aggregate_stock(aisles)
        selected_orders, demand, total_units = self._pick_orders(
            indices, orders, order_sizes, stock, ub
        )

        if total_units < lb:
            return dict(_EMPTY_RESULT)

        visited_aisles = (
            multi_greedy_aisle_select(demand, aisles)
            if self._greedy == "multi"
            else greedy_aisle_select(demand, aisles)
        )
        if not visited_aisles:
            return dict(_EMPTY_RESULT)

        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": total_units / len(visited_aisles),
        }

    def _build_traversal(self, n_orders: int, order_sizes: list[int]) -> list[int]:
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
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock

    @staticmethod
    def _pick_orders(
        indices: list[int],
        orders: list[dict[int, int]],
        order_sizes: list[int],
        stock: dict[int, int],
        ub: int,
    ) -> tuple[list[int], dict[int, int], int]:
        selected: list[int] = []
        demand: dict[int, int] = {}
        total = 0

        for idx in indices:
            size = order_sizes[idx]
            if total + size > ub:
                continue

            order = orders[idx]
            if any(stock.get(item, 0) < qty for item, qty in order.items()):
                continue

            selected.append(idx)
            total += size
            for item, qty in order.items():
                stock[item] -= qty
                demand[item] = demand.get(item, 0) + qty

        return selected, demand, total
