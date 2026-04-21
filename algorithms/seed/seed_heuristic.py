import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.similarity import similarity
from problems.base import ProblemInput

_VALID_SEED_STRATEGY = {"biggest", "smallest", "most_shared", "random"}
_VALID_SYNERGY = {"min_new_aisles", "max_similarity"}
_VALID_GREEDY = {"simple", "multi"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class SeedHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)
        seed_strategy = params.get("seed_strategy")
        if seed_strategy not in _VALID_SEED_STRATEGY:
            raise ValueError(
                f"SeedHeuristic: invalid 'seed_strategy'={seed_strategy!r}; "
                f"expected one of {sorted(_VALID_SEED_STRATEGY)}"
            )
        synergy = params.get("synergy")
        if synergy not in _VALID_SYNERGY:
            raise ValueError(
                f"SeedHeuristic: invalid 'synergy'={synergy!r}; "
                f"expected one of {sorted(_VALID_SYNERGY)}"
            )
        greedy = params.get("greedy")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"SeedHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )
        similarity_weighted = params.get("similarity_weighted", False)
        if not isinstance(similarity_weighted, bool):
            raise ValueError(
                f"SeedHeuristic: invalid 'similarity_weighted'={similarity_weighted!r}; "
                f"expected bool"
            )
        self._seed_strategy = seed_strategy
        self._synergy = synergy
        self._greedy = greedy
        self._similarity_weighted = similarity_weighted
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "seed_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders
        n_aisles = instance.nAisles

        if n_orders == 0 or n_aisles == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in orders]
        stock = self._aggregate_stock(aisles)

        seed_idx = self._pick_seed(orders, aisles, order_sizes)
        if seed_idx is None:
            return dict(_EMPTY_RESULT)

        seed_order = orders[seed_idx]
        seed_size = order_sizes[seed_idx]
        if seed_size > ub:
            return dict(_EMPTY_RESULT)
        if any(stock.get(it, 0) < q for it, q in seed_order.items()):
            return dict(_EMPTY_RESULT)

        selected = [seed_idx]
        demand: dict[int, int] = dict(seed_order)
        total_units = seed_size
        stock_remaining = dict(stock)
        for it, q in seed_order.items():
            stock_remaining[it] -= q

        remaining = set(range(n_orders)) - {seed_idx}

        cached_aisles_now: set[int] | None = None

        while remaining:
            candidates: list[int] = []
            for idx in remaining:
                if total_units + order_sizes[idx] > ub:
                    continue
                order = orders[idx]
                if any(stock_remaining.get(it, 0) < q for it, q in order.items()):
                    continue
                candidates.append(idx)

            if not candidates:
                break

            if self._synergy == "max_similarity":
                best = max(
                    candidates,
                    key=lambda i: (
                        similarity(
                            demand, orders[i], weighted=self._similarity_weighted
                        ),
                        order_sizes[i],
                    ),
                )
            else:
                if cached_aisles_now is None:
                    cached_aisles_now = set(greedy_aisle_select(dict(demand), aisles))
                aisles_now = cached_aisles_now

                def new_aisles_count(i: int) -> int:
                    combined = dict(demand)
                    for it, q in orders[i].items():
                        combined[it] = combined.get(it, 0) + q
                    after = set(greedy_aisle_select(combined, aisles))
                    return len(after - aisles_now)

                best = min(
                    candidates,
                    key=lambda i: (new_aisles_count(i), -order_sizes[i]),
                )

            selected.append(best)
            best_order = orders[best]
            total_units += order_sizes[best]
            for it, q in best_order.items():
                demand[it] = demand.get(it, 0) + q
                stock_remaining[it] -= q
            remaining.remove(best)
            cached_aisles_now = None

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

    def _pick_seed(
        self,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_sizes: list[int],
    ) -> int | None:
        n = len(orders)
        non_empty = [i for i in range(n) if order_sizes[i] > 0]
        if not non_empty:
            return None

        if self._seed_strategy == "biggest":
            return max(non_empty, key=order_sizes.__getitem__)
        if self._seed_strategy == "smallest":
            return min(non_empty, key=order_sizes.__getitem__)
        if self._seed_strategy == "random":
            rng = random.Random(self._seed) if self._seed is not None else random
            return rng.choice(non_empty)

        item_aisle_count: dict[int, int] = {}
        for aisle in aisles:
            for item in aisle:
                item_aisle_count[item] = item_aisle_count.get(item, 0) + 1
        return max(
            non_empty,
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
