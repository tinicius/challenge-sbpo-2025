from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput

_VALID_GREEDY = {"simple", "multi"}
_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class SeedHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)
        greedy = params.get("greedy", "simple")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"SeedHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )
        self._greedy = greedy

    @property
    def name(self) -> str:
        return "seed_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders

        if n_orders == 0 or instance.nAisles == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in orders]
        stock = self._aggregate_stock(aisles)

        valid_orders = [
            i for i in range(n_orders)
            if order_sizes[i] > 0
            and order_sizes[i] <= ub
            and all(stock.get(it, 0) >= q for it, q in orders[i].items())
        ]

        if not valid_orders:
            return dict(_EMPTY_RESULT)

        # Precompute aisle sets for valid orders (optimization for large instances)
        order_aisles = {}
        for idx in valid_orders:
            order_aisles[idx] = set(greedy_aisle_select(dict(orders[idx]), aisles))

        seed_idx = max(
            valid_orders,
            key=lambda i: (len(order_aisles[i]), order_sizes[i]),
        )

        seed_order = orders[seed_idx]
        seed_size = order_sizes[seed_idx]

        selected = [seed_idx]
        demand: dict[int, int] = dict(seed_order)
        total_units = seed_size
        stock_remaining = dict(stock)

        for it, q in seed_order.items():
            stock_remaining[it] -= q

        remaining = set(valid_orders) - {seed_idx}
        aisles_now = set(greedy_aisle_select(dict(demand), aisles))

        while remaining:
            candidates = []
            for idx in remaining:
                if total_units + order_sizes[idx] > ub:
                    continue
                order = orders[idx]
                if any(stock_remaining.get(it, 0) < q for it, q in order.items()):
                    continue
                new_aisles = len(order_aisles[idx] - aisles_now)
                candidates.append((idx, new_aisles))

            if not candidates:
                break

            best_idx, _ = min(
                candidates,
                key=lambda x: (x[1], -order_sizes[x[0]]),
            )

            selected.append(best_idx)
            best_order = orders[best_idx]
            total_units += order_sizes[best_idx]
            for it, q in best_order.items():
                demand[it] = demand.get(it, 0) + q
                stock_remaining[it] -= q
            remaining.remove(best_idx)
            aisles_now = aisles_now | order_aisles[best_idx]

        if total_units < lb:
            return dict(_EMPTY_RESULT)

        visited_aisles = (
            multi_greedy_aisle_select(demand, aisles)
            if self._greedy == "multi"
            else greedy_aisle_select(dict(demand), aisles)
        )

        if not visited_aisles:
            return dict(_EMPTY_RESULT)

        return {
            "selected_orders": selected,
            "visited_aisles": visited_aisles,
            "objective": total_units / len(visited_aisles),
        }

    @staticmethod
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
