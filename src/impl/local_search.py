import random

from models.solver import Solver
from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select


class LocalSearchHeuristic(Solver):
    """
    Hybrid heuristic: greedy order-centric construction followed by
    hill-climbing local search.

    Accepted moves (first-improvement within each iteration):
      - Add:    insert one order not yet in the wave (if still feasible).
      - Remove: drop one order from the wave (if still feasible).
      - Swap:   replace one in-wave order with one outside-wave order.

    After every accepted move the minimum aisle set is recomputed with a
    greedy set-cover and the objective (units / aisles) is re-evaluated.

    Config keys
    -----------
    iterations      : int  – number of independent restarts (default 1).
    max_no_improve  : int  – consecutive non-improving full passes before
                             stopping (default 5).
    random_seed     : any  – seed for the internal RNG (default None).
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
    # local search                                                         #
    # ------------------------------------------------------------------ #

    def _local_search(
        self,
        initial: list[int],
        rng: random.Random,
        max_no_improve: int,
    ) -> tuple[list[int], list[int]]:
        """Hill-climbing: iterate over add / remove / swap moves until no
        improvement is found for ``max_no_improve`` consecutive passes."""
        current_set = set(initial)
        current_aisles = self._min_aisles(initial)
        current_obj = self._objective(initial, current_aisles)

        no_improve = 0
        while no_improve < max_no_improve:
            current_list = list(current_set)
            outside_list = list(set(range(self.n_orders)) - current_set)
            rng.shuffle(current_list)
            rng.shuffle(outside_list)

            improved = False

            # --- Add move ---
            for o_add in outside_list:
                candidate = current_list + [o_add]
                if not self._is_feasible(candidate):
                    continue
                new_aisles = self._min_aisles(candidate)
                new_obj = self._objective(candidate, new_aisles)
                if new_obj > current_obj:
                    current_set = set(candidate)
                    current_aisles = new_aisles
                    current_obj = new_obj
                    improved = True
                    break

            if improved:
                no_improve = 0
                continue

            # --- Remove move ---
            for o_rem in current_list:
                candidate = [o for o in current_list if o != o_rem]
                if not self._is_feasible(candidate):
                    continue
                new_aisles = self._min_aisles(candidate)
                new_obj = self._objective(candidate, new_aisles)
                if new_obj > current_obj:
                    current_set = set(candidate)
                    current_aisles = new_aisles
                    current_obj = new_obj
                    improved = True
                    break

            if improved:
                no_improve = 0
                continue

            # --- Swap move ---
            for o_rem in current_list:
                for o_add in outside_list:
                    candidate = [o for o in current_list if o != o_rem] + [o_add]
                    if not self._is_feasible(candidate):
                        continue
                    new_aisles = self._min_aisles(candidate)
                    new_obj = self._objective(candidate, new_aisles)
                    if new_obj > current_obj:
                        current_set = set(candidate)
                        current_aisles = new_aisles
                        current_obj = new_obj
                        improved = True
                        break
                if improved:
                    break

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return list(current_set), current_aisles

    # ------------------------------------------------------------------ #
    # solve                                                                #
    # ------------------------------------------------------------------ #

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self.config.get("random_seed", None)
        rng = random.Random(seed)
        max_no_improve: int = self.config.get("max_no_improve", 5)
        iterations: int = self.config.get("iterations", 1)

        self._total_stock = self._build_total_stock()

        all_orders = list(range(self.n_orders))

        best_obj = -1.0
        best_orders: list[int] = []
        best_aisles: list[int] = []

        for _ in range(iterations):
            order_seq = list(all_orders)
            rng.shuffle(order_seq)

            initial = self._greedy_initial(order_seq)
            if self._total_units(initial) < self.lb:
                continue

            orders, aisles = self._local_search(initial, rng, max_no_improve)

            if not aisles or not self._is_feasible(orders):
                continue

            obj = self._objective(orders, aisles)
            if obj > best_obj:
                best_obj = obj
                best_orders = orders
                best_aisles = aisles

        if best_obj < 0:
            return [], []

        return best_orders, best_aisles
