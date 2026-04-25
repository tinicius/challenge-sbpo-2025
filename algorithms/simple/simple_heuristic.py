import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from algorithms.utils.similarity import similarity
from problems.base import ProblemInput

_VALID_ORDER = {None, "asc", "desc", "similar", "diff"}
_VALID_GREEDY = {"simple", "multi"}
_VALID_FIRST_ORDER = {None, "smaller", "bigger"}

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
        if first_order not in _VALID_FIRST_ORDER:
            raise ValueError(
                f"SimpleHeuristic: invalid 'first_order'={first_order!r}; "
                f"expected one of {sorted(v for v in _VALID_FIRST_ORDER if v)} or unset"
            )

        similarity_weighted = params.get("similarity_weighted", False)
        if not isinstance(similarity_weighted, bool):
            raise ValueError(
                f"SimpleHeuristic: invalid 'similarity_weighted'={similarity_weighted!r}; "
                "expected bool"
            )

        exact = params.get("exact", False)
        if not isinstance(exact, bool):
            raise ValueError(
                f"SimpleHeuristic: invalid 'exact'={exact!r}; expected bool"
            )

        gurobi_time_limit = params.get("gurobi_time_limit", 30.0)
        gurobi_threads = params.get("gurobi_threads", 0)

        try:
            self._gurobi_time_limit = float(gurobi_time_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SimpleHeuristic: 'gurobi_time_limit' must be a number"
            ) from exc
        if self._gurobi_time_limit <= 0:
            raise ValueError("SimpleHeuristic: 'gurobi_time_limit' must be > 0")

        try:
            self._gurobi_threads = int(gurobi_threads)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SimpleHeuristic: 'gurobi_threads' must be an integer"
            ) from exc
        if self._gurobi_threads < 0:
            raise ValueError("SimpleHeuristic: 'gurobi_threads' must be >= 0")

        self._order = order
        self._greedy = greedy
        self._first_order = first_order
        self._similarity_weighted = similarity_weighted
        self._exact = exact
        self._seed = params.get("seed")

    @property
    def name(self) -> str:
        return "simple_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub

        order_sizes = [sum(o.values()) for o in orders]
        indices = self._build_traversal(instance.nOrders, order_sizes, orders)

        stock = self._aggregate_stock(aisles)
        selected_orders, demand, total_units = self._pick_orders(
            indices, orders, order_sizes, stock, ub
        )

        if total_units < lb:
            return dict(_EMPTY_RESULT)

        if self._exact:
            visited_aisles = self._solve_min_aisles_for_demand(demand, aisles)
        else:
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

    def _solve_min_aisles_for_demand(
        self,
        demand: dict[int, int],
        aisles: list[dict[int, int]],
    ) -> list[int]:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise RuntimeError(
                "SimpleHeuristic with exact=True requires gurobipy to be installed and licensed."
            ) from exc

        active_demand = {item: qty for item, qty in demand.items() if qty > 0}
        if not active_demand:
            return []

        model = gp.Model("simple_min_aisles")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self._gurobi_time_limit
        if self._gurobi_threads > 0:
            model.Params.Threads = self._gurobi_threads

        n_aisles = len(aisles)
        x = model.addVars(n_aisles, vtype=GRB.BINARY, name="x")

        model.setObjective(gp.quicksum(x[a] for a in range(n_aisles)), GRB.MINIMIZE)

        for item, qty in active_demand.items():
            model.addConstr(
                gp.quicksum(aisles[a].get(item, 0) * x[a] for a in range(n_aisles))
                >= qty,
                name=f"cover_item_{item}",
            )

        model.optimize()

        if (
            model.Status in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}
            and model.SolCount > 0
        ):
            return [a for a in range(n_aisles) if x[a].X >= 0.5]

        return []

    def _build_traversal(
        self,
        n_orders: int,
        order_sizes: list[int],
        orders: list[dict[int, int]],
    ) -> list[int]:
        if self._order is None:
            rng = random.Random(self._seed) if self._seed is not None else random
            indices = list(range(n_orders))
            rng.shuffle(indices)
            return indices
        if self._order in {"similar", "diff"}:
            reference = self._pick_first_order(n_orders, order_sizes, orders)
            return sorted(
                range(n_orders),
                key=lambda i: similarity(
                    reference, orders[i], weighted=self._similarity_weighted
                ),
                reverse=(self._order == "similar"),
            )
        return sorted(
            range(n_orders),
            key=order_sizes.__getitem__,
            reverse=(self._order == "desc"),
        )

    def _pick_first_order(
        self,
        n_orders: int,
        order_sizes: list[int],
        orders: list[dict[int, int]],
    ) -> dict[int, int]:
        if self._first_order == "bigger":
            return orders[max(range(n_orders), key=order_sizes.__getitem__)]
        if self._first_order == "smaller":
            return orders[min(range(n_orders), key=order_sizes.__getitem__)]
        rng = random.Random(self._seed) if self._seed is not None else random
        indices = list(range(n_orders))
        rng.shuffle(indices)
        return orders[indices[0]]

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
