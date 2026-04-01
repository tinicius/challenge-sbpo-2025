from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class MinAisleCover(Algorithm):
    """
    Minimum-aisle-cover post-order-fix heuristic.

    Phase (a): fix order set O' by greedily packing orders (largest-first)
               until the wave upper bound is reached.
    Phase (b): given the demand from O', solve minimum set cover over
               demanded items using the greedy set-cover heuristic
               (ln|I|-approximation).

    The algorithm iterates over several order-fill levels between lb and ub,
    keeping the best items/aisles ratio found.
    """

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "min_aisle_cover"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders = inst.orders
        aisles = inst.aisles
        lb = inst.lb
        ub = inst.ub
        n_orders = inst.nOrders

        order_units = [sum(order.values()) for order in orders]

        # Build multiple order sequences to try
        order_sequences = self._build_order_sequences(n_orders, orders, order_units)

        best_obj = -1.0
        best_orders: list[int] = []
        best_aisles: list[int] = []

        for sequence in order_sequences:
            self._try_sequence(
                sequence, orders, aisles, order_units, lb, ub,
                best_obj, best_orders, best_aisles,
            )
            # Update best from returned values
            obj, ords, ais = self._try_sequence(
                sequence, orders, aisles, order_units, lb, ub,
                best_obj, best_orders, best_aisles,
            )
            if obj > best_obj:
                best_obj = obj
                best_orders = ords
                best_aisles = ais

        if not best_orders or not best_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        total_items = sum(order_units[o] for o in best_orders)
        objective = total_items / len(best_aisles)
        return {
            "selected_orders": best_orders,
            "visited_aisles": best_aisles,
            "objective": objective,
        }

    def _build_order_sequences(
        self,
        n_orders: int,
        orders: list[dict[int, int]],
        order_units: list[int],
    ) -> list[list[int]]:
        indices = list(range(n_orders))

        # Largest orders first (pack more items per order → fewer aisles needed)
        desc = sorted(indices, key=lambda i: -order_units[i])

        # Fewest distinct items first (concentrated demand → fewer aisles)
        few_items = sorted(indices, key=lambda i: (len(orders[i]), -order_units[i]))

        # Highest density (units / distinct items) first
        density = sorted(
            indices,
            key=lambda i: -order_units[i] / max(len(orders[i]), 1),
        )

        return [desc, few_items, density]

    def _try_sequence(
        self,
        sequence: list[int],
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        order_units: list[int],
        lb: int,
        ub: int,
        current_best_obj: float,
        current_best_orders: list[int],
        current_best_aisles: list[int],
    ) -> tuple[float, list[int], list[int]]:
        best_obj = current_best_obj
        best_orders = list(current_best_orders)
        best_aisles = list(current_best_aisles)

        # Phase (a): greedily pack orders following the sequence
        selected: list[int] = []
        demand: dict[int, int] = {}
        total_units = 0

        for order_idx in sequence:
            units = order_units[order_idx]
            if total_units + units > ub:
                continue

            selected.append(order_idx)
            total_units += units
            for item, qty in orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

            # Only evaluate once we reach the lower bound
            if total_units < lb:
                continue

            # Phase (b): greedy minimum set cover for current demand
            covered_aisles = multi_greedy_aisle_select(
                {item: qty for item, qty in demand.items() if qty > 0},
                aisles,
            )

            if not covered_aisles:
                continue

            # Verify full coverage
            if not self._is_fully_covered(demand, aisles, covered_aisles):
                continue

            obj = total_units / len(covered_aisles)
            if obj > best_obj:
                best_obj = obj
                best_orders = list(selected)
                best_aisles = covered_aisles

        return best_obj, best_orders, best_aisles

    @staticmethod
    def _is_fully_covered(
        demand: dict[int, int],
        aisles: list[dict[int, int]],
        aisle_indices: list[int],
    ) -> bool:
        supply: dict[int, int] = {}
        for idx in aisle_indices:
            for item, qty in aisles[idx].items():
                supply[item] = supply.get(item, 0) + qty
        for item, qty in demand.items():
            if qty > 0 and supply.get(item, 0) < qty:
                return False
        return True
