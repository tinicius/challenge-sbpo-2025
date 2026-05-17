import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.ilp_aisle_select import solve_min_aisle_cover
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.similarity import similarity
from problems.base import ProblemInput

_VALID_ORDER = {
    "random",
    "asc",
    "desc",
    "similar",
    "similar_weighted",
    "diff",
    "diff_weighted",
}

_VALID_GREEDY = {"simple", "multi", "exact"}

_VALID_FIRST_ORDER = {"random", "smaller", "bigger", "most_shared"}

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

        first_order = params.get("first_order")
        if first_order not in _VALID_FIRST_ORDER and order in {
            "similar",
            "similar_weighted",
            "diff",
            "diff_weighted",
        }:
            raise ValueError(
                f"SimpleHeuristic: invalid 'first_order'={first_order!r}; "
                f"expected one of {sorted(v for v in _VALID_FIRST_ORDER if v)} or unset"
            )

        exact_time_limit = params.get("exact_time_limit", 30.0)
        exact_num_workers = params.get("exact_num_workers", 0)

        try:
            self._exact_time_limit = float(exact_time_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SimpleHeuristic: 'exact_time_limit' must be a number"
            ) from exc
        if self._exact_time_limit <= 0:
            raise ValueError("SimpleHeuristic: 'exact_time_limit' must be > 0")

        try:
            self._exact_num_workers = int(exact_num_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SimpleHeuristic: 'exact_num_workers' must be an integer"
            ) from exc
        if self._exact_num_workers < 0:
            raise ValueError("SimpleHeuristic: 'exact_num_workers' must be >= 0")

        self._order = order
        self._greedy = greedy
        self._first_order = first_order
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "simple_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub

        order_sizes = [sum(order.values()) for order in orders]

        stock = self._aggregate_stock(aisles)

        sequence = self._build_sequence(instance.nOrders, order_sizes, orders, aisles)

        selected_orders, demand, total_units = self._pick_orders(
            sequence, orders, order_sizes, stock, ub
        )

        if total_units < lb:
            return dict(_EMPTY_RESULT)

        match self._greedy:
            case "exact":
                visited_aisles = solve_min_aisle_cover(
                    demand,
                    aisles,
                    time_limit_seconds=self._exact_time_limit,
                    num_workers=self._exact_num_workers,
                ).selected_aisles
            case "multi":
                visited_aisles = multi_greedy_aisle_select(demand, aisles)
            case "simple":
                visited_aisles = greedy_aisle_select(demand, aisles)

        if not visited_aisles:
            return dict(_EMPTY_RESULT)

        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": total_units / len(visited_aisles),
        }

    def _build_sequence(
        self,
        n_orders: int,
        order_sizes: list[int],
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
    ) -> list[int]:

        indices = list(range(n_orders))

        match self._order:
            case "random":
                rng = random.Random(self._seed) if self._seed is not None else random
                rng.shuffle(indices)
                return indices
            case "similar" | "similar_weighted" | "diff" | "diff_weighted":
                reference = self._pick_first_order(
                    n_orders, order_sizes, orders, aisles
                )

                weighted = (
                    self._order == "similar_weighted" or self._order == "diff_weighted"
                )

                similar = self._order == "similar" or self._order == "similar_weighted"

                return sorted(
                    indices,
                    key=lambda i: similarity(
                        reference,
                        orders[i],
                        weighted=weighted,
                    ),
                    reverse=similar,
                )
            case "asc":
                return sorted(indices, key=order_sizes.__getitem__)
            case "desc":
                return sorted(indices, key=order_sizes.__getitem__, reverse=True)

        raise ValueError(f"SimpleHeuristic: invalid 'order'={self._order!r}")

    def _pick_first_order(
        self,
        n_orders: int,
        order_sizes: list[int],
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
    ) -> dict[int, int]:
        if self._first_order == "bigger":
            return orders[max(range(n_orders), key=order_sizes.__getitem__)]
        if self._first_order == "smaller":
            return orders[min(range(n_orders), key=order_sizes.__getitem__)]
        if self._first_order == "most_shared":
            return orders[
                self._get_most_shared_order(n_orders, aisles, order_sizes, orders)
            ]

        rng = random.Random(self._seed) if self._seed is not None else random
        indices = list(range(n_orders))
        rng.shuffle(indices)
        return orders[indices[0]]

    def _get_most_shared_order(self, n_orders, aisles, order_sizes, orders):
        item_aisle_count: dict[int, int] = {}

        for aisle in aisles:
            for item in aisle:
                item_aisle_count[item] = item_aisle_count.get(item, 0) + 1

        return max(
            range(n_orders),
            key=lambda i: (
                sum(item_aisle_count.get(it, 0) for it in orders[i]),
                order_sizes[i],
            ),
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
