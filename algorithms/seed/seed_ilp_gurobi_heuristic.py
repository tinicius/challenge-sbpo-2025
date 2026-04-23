from __future__ import annotations

from algorithms.base import Algorithm
from problems.base import ProblemInput

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
_VALID_SEED_STRATEGY = {
    "most_distinct_items",
    "least_distinct_items",
    "most_total_items",
    "least_total_items",
}


class SeedIlpGurobiHeuristic(Algorithm):
    def __init__(self, params: dict):
        super().__init__(params)

        seed_strategy = params.get("seed_strategy")
        if seed_strategy not in _VALID_SEED_STRATEGY:
            raise ValueError(
                "SeedIlpGurobiHeuristic: invalid 'seed_strategy'="
                f"{seed_strategy!r}; expected one of {sorted(_VALID_SEED_STRATEGY)}"
            )

        gurobi_time_limit = params.get("gurobi_time_limit", 30.0)
        gurobi_threads = params.get("gurobi_threads", 0)
        seed = params.get("seed")

        try:
            self._gurobi_time_limit = float(gurobi_time_limit)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SeedIlpGurobiHeuristic: 'gurobi_time_limit' must be a number"
            ) from exc
        if self._gurobi_time_limit <= 0:
            raise ValueError("SeedIlpGurobiHeuristic: 'gurobi_time_limit' must be > 0")

        try:
            self._gurobi_threads = int(gurobi_threads)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SeedIlpGurobiHeuristic: 'gurobi_threads' must be an integer"
            ) from exc
        if self._gurobi_threads < 0:
            raise ValueError("SeedIlpGurobiHeuristic: 'gurobi_threads' must be >= 0")

        if seed is not None:
            try:
                int(seed)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "SeedIlpGurobiHeuristic: 'seed' must be an integer when provided"
                ) from exc

        self._seed_strategy = seed_strategy

    @property
    def name(self) -> str:
        return "seed_ilp_gurobi_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders = inst.orders
        aisles = inst.aisles
        n_orders = inst.nOrders

        if n_orders == 0 or inst.nAisles == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(order.values()) for order in orders]
        seed_candidates = self._rank_seed_candidates(orders, order_sizes)

        seed_idx = None
        visited_aisles = None
        inventory_pool = None

        for candidate_idx in seed_candidates:
            candidate_order = orders[candidate_idx]
            candidate_size = order_sizes[candidate_idx]
            if candidate_size > inst.ub:
                continue

            candidate_aisles = self._solve_min_aisles_for_demand(
                candidate_order, aisles
            )
            if not candidate_aisles:
                continue

            candidate_pool = self._build_inventory_pool(candidate_aisles, aisles)
            if any(
                candidate_pool.get(item, 0) < qty
                for item, qty in candidate_order.items()
            ):
                continue

            seed_idx = candidate_idx
            visited_aisles = candidate_aisles
            inventory_pool = candidate_pool
            break

        if seed_idx is None or visited_aisles is None or inventory_pool is None:
            return dict(_EMPTY_RESULT)

        seed_order = orders[seed_idx]
        seed_size = order_sizes[seed_idx]

        selected_orders = [seed_idx]
        selected_set = {seed_idx}
        wave_demand = dict(seed_order)
        total_units = seed_size

        ranked_candidates = sorted(
            (
                idx
                for idx in range(n_orders)
                if idx != seed_idx and order_sizes[idx] > 0
            ),
            key=lambda idx: (-order_sizes[idx], idx),
        )

        selected_orders, total_units = self._greedy_absorb_orders(
            selected_orders=selected_orders,
            selected_set=selected_set,
            ranked_candidates=ranked_candidates,
            orders=orders,
            order_sizes=order_sizes,
            wave_demand=wave_demand,
            inventory_pool=inventory_pool,
            total_units=total_units,
            ub=inst.ub,
        )

        visited_set = set(visited_aisles)
        while total_units < inst.lb:
            best_aisle = None
            best_gain = 0

            for aisle_idx, aisle in enumerate(aisles):
                if aisle_idx in visited_set:
                    continue

                aisle_gain = self._unlocked_items_gain(
                    candidate_aisle=aisle,
                    selected_set=selected_set,
                    ranked_candidates=ranked_candidates,
                    orders=orders,
                    order_sizes=order_sizes,
                    wave_demand=wave_demand,
                    inventory_pool=inventory_pool,
                    total_units=total_units,
                    ub=inst.ub,
                )

                if aisle_gain > best_gain:
                    best_gain = aisle_gain
                    best_aisle = aisle_idx

            if best_aisle is None or best_gain <= 0:
                break

            visited_aisles.append(best_aisle)
            visited_set.add(best_aisle)
            self._add_aisle_to_pool(inventory_pool, aisles[best_aisle])

            selected_orders, total_units = self._greedy_absorb_orders(
                selected_orders=selected_orders,
                selected_set=selected_set,
                ranked_candidates=ranked_candidates,
                orders=orders,
                order_sizes=order_sizes,
                wave_demand=wave_demand,
                inventory_pool=inventory_pool,
                total_units=total_units,
                ub=inst.ub,
            )

        if total_units < inst.lb:
            return dict(_EMPTY_RESULT)

        return {
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "objective": total_units / len(visited_aisles),
        }

    def _rank_seed_candidates(
        self,
        orders: list[dict[int, int]],
        order_sizes: list[int],
    ) -> list[int]:
        candidates = [idx for idx, size in enumerate(order_sizes) if size > 0]
        if not candidates:
            return []

        if self._seed_strategy == "most_distinct_items":
            return sorted(candidates, key=lambda idx: (-len(orders[idx]), idx))
        if self._seed_strategy == "least_distinct_items":
            return sorted(candidates, key=lambda idx: (len(orders[idx]), idx))
        if self._seed_strategy == "most_total_items":
            return sorted(candidates, key=lambda idx: (-order_sizes[idx], idx))
        return sorted(candidates, key=lambda idx: (order_sizes[idx], idx))

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
                "SeedIlpGurobiHeuristic requires gurobipy to be installed and licensed."
            ) from exc

        active_demand = {item: qty for item, qty in demand.items() if qty > 0}
        if not active_demand:
            return []

        model = gp.Model("seed_min_aisles")
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

    @staticmethod
    def _build_inventory_pool(
        visited_aisles: list[int],
        aisles: list[dict[int, int]],
    ) -> dict[int, int]:
        pool: dict[int, int] = {}
        for aisle_idx in visited_aisles:
            for item, qty in aisles[aisle_idx].items():
                pool[item] = pool.get(item, 0) + qty
        return pool

    @staticmethod
    def _can_add_order(
        order: dict[int, int],
        current_demand: dict[int, int],
        inventory_pool: dict[int, int],
    ) -> bool:
        for item, qty in order.items():
            if current_demand.get(item, 0) + qty > inventory_pool.get(item, 0):
                return False
        return True

    def _greedy_absorb_orders(
        self,
        selected_orders: list[int],
        selected_set: set[int],
        ranked_candidates: list[int],
        orders: list[dict[int, int]],
        order_sizes: list[int],
        wave_demand: dict[int, int],
        inventory_pool: dict[int, int],
        total_units: int,
        ub: int,
    ) -> tuple[list[int], int]:
        for idx in ranked_candidates:
            if idx in selected_set:
                continue

            if total_units + order_sizes[idx] > ub:
                continue

            order = orders[idx]
            if not self._can_add_order(order, wave_demand, inventory_pool):
                continue

            selected_orders.append(idx)
            selected_set.add(idx)
            total_units += order_sizes[idx]
            for item, qty in order.items():
                wave_demand[item] = wave_demand.get(item, 0) + qty

        return selected_orders, total_units

    def _unlocked_items_gain(
        self,
        candidate_aisle: dict[int, int],
        selected_set: set[int],
        ranked_candidates: list[int],
        orders: list[dict[int, int]],
        order_sizes: list[int],
        wave_demand: dict[int, int],
        inventory_pool: dict[int, int],
        total_units: int,
        ub: int,
    ) -> int:
        gain = 0
        for idx in ranked_candidates:
            if idx in selected_set:
                continue

            if total_units + order_sizes[idx] > ub:
                continue

            order = orders[idx]
            feasible = True
            for item, qty in order.items():
                pool_qty = inventory_pool.get(item, 0) + candidate_aisle.get(item, 0)
                if wave_demand.get(item, 0) + qty > pool_qty:
                    feasible = False
                    break

            if feasible:
                gain += order_sizes[idx]

        return gain

    @staticmethod
    def _add_aisle_to_pool(
        inventory_pool: dict[int, int],
        aisle: dict[int, int],
    ) -> None:
        for item, qty in aisle.items():
            inventory_pool[item] = inventory_pool.get(item, 0) + qty
