import math
import random

from models.solver import Solver
from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select


class SimulatedAnnealingHeuristic(Solver):
    """
    Metaheuristic: Simulated Annealing for the Wave Order-Picking problem.

    State        : set of orders currently in the wave.
    Neighbourhood: add / remove / swap one order (chosen at random).
    Acceptance   : Metropolis criterion  P(accept worse) = exp(Δobj / T).
    Cooling      : geometric schedule    T_{k+1} = alpha · T_k.

    Config keys
    -----------
    T_init      : float – initial temperature (default 5.0).
    T_min       : float – stopping temperature (default 0.01).
    alpha       : float – cooling rate in (0, 1) (default 0.995).
    max_iter    : int   – maximum number of moves (default 5000).
    random_seed : any   – seed for the internal RNG (default None).
    """

    # ------------------------------------------------------------------ #
    # helpers                                                              #
    # ------------------------------------------------------------------ #

    def _build_total_stock(self) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in self.aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock

    def _total_units(self, orders: list[int]) -> int:
        return sum(sum(self.orders[o].values()) for o in orders)

    def _build_demand(self, orders: list[int]) -> dict[int, int]:
        demand: dict[int, int] = {}
        for o in orders:
            for item, qty in self.orders[o].items():
                demand[item] = demand.get(item, 0) + qty
        return demand

    def _min_aisles(self, orders: list[int]) -> list[int]:
        demand = self._build_demand(orders)
        return multi_greedy_aisle_select(demand, self.aisles)

    def _objective(self, orders: list[int], aisles: list[int]) -> float:
        if not aisles:
            return 0.0
        return self._total_units(orders) / len(aisles)

    def _is_feasible(self, orders: list[int]) -> bool:
        total = self._total_units(orders)
        if not (self.lb <= total <= self.ub):
            return False
        demand = self._build_demand(orders)
        for item, qty in demand.items():
            if self._total_stock.get(item, 0) < qty:
                return False
        return True

    # ------------------------------------------------------------------ #
    # initial construction                                                 #
    # ------------------------------------------------------------------ #

    def _greedy_initial(self, order_sequence: list[int]) -> list[int]:
        """Order-centric greedy: accept orders while total ≤ UB and stock allows."""
        remaining_stock = dict(self._total_stock)
        selected: list[int] = []
        total = 0
        for o in order_sequence:
            order = self.orders[o]
            units = sum(order.values())
            if total + units > self.ub:
                continue
            if any(remaining_stock.get(item, 0) < qty for item, qty in order.items()):
                continue
            selected.append(o)
            total += units
            for item, qty in order.items():
                remaining_stock[item] -= qty
        return selected

    # ------------------------------------------------------------------ #
    # neighbourhood                                                        #
    # ------------------------------------------------------------------ #

    def _random_neighbor(
        self, current: list[int], rng: random.Random
    ) -> list[int] | None:
        """Return a neighbouring solution by applying one random move."""
        current_set = set(current)
        outside = list(set(range(self.n_orders)) - current_set)

        move = rng.choice(["add", "remove", "swap"])

        if move == "add" and outside:
            return current + [rng.choice(outside)]

        if move == "remove" and len(current) > 1:
            o_rem = rng.choice(current)
            return [o for o in current if o != o_rem]

        if move == "swap" and current and outside:
            o_out = rng.choice(current)
            o_in = rng.choice(outside)
            return [o for o in current if o != o_out] + [o_in]

        return None

    # ------------------------------------------------------------------ #
    # solve                                                                #
    # ------------------------------------------------------------------ #

    def solve(self) -> tuple[list[int], list[int]]:
        T_init: float = self.config.get("T_init", 5.0)
        T_min: float = self.config.get("T_min", 0.01)
        alpha: float = self.config.get("alpha", 0.995)
        max_iter: int = self.config.get("max_iter", 5000)
        seed = self.config.get("random_seed", None)
        rng = random.Random(seed)

        self._total_stock = self._build_total_stock()

        all_orders = list(range(self.n_orders))
        rng.shuffle(all_orders)
        current = self._greedy_initial(all_orders)

        if self._total_units(current) < self.lb:
            return [], []

        current_aisles = self._min_aisles(current)
        current_obj = self._objective(current, current_aisles)

        best_orders = list(current)
        best_aisles = list(current_aisles)
        best_obj = current_obj

        T = T_init
        i = 0

        while T > T_min and i < max_iter:
            neighbor = self._random_neighbor(current, rng)

            if neighbor is None or not self._is_feasible(neighbor):
                i += 1
                continue

            neighbor_aisles = self._min_aisles(neighbor)
            neighbor_obj = self._objective(neighbor, neighbor_aisles)

            delta = neighbor_obj - current_obj

            if delta > 0 or rng.random() < math.exp(delta / T):
                current = neighbor
                current_aisles = neighbor_aisles
                current_obj = neighbor_obj

                if current_obj > best_obj:
                    best_obj = current_obj
                    best_orders = list(current)
                    best_aisles = list(current_aisles)

            T *= alpha
            i += 1

        if best_obj <= 0 or not best_orders:
            return [], []

        return best_orders, best_aisles
