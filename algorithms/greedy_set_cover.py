from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class GreedySetCover(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "greedy_set_cover"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders = inst.orders
        aisles = inst.aisles
        lb = inst.lb
        ub = inst.ub
        n_orders = inst.nOrders

        order_units = [sum(order.values()) for order in orders]

        selected_orders: list[int] = []
        current_demand: dict[int, int] = {}
        total_units = 0

        best_obj = -1.0
        best_orders: list[int] = []
        best_aisles: list[int] = []

        remaining_candidates = set(range(n_orders))

        while remaining_candidates:
            best_candidate = -1
            best_candidate_ratio = -1.0
            best_candidate_aisles: list[int] = []

            for candidate in remaining_candidates:
                candidate_units = order_units[candidate]
                tentative_units = total_units + candidate_units
                if tentative_units > ub:
                    continue

                tentative_demand = dict(current_demand)
                for item, qty in orders[candidate].items():
                    tentative_demand[item] = tentative_demand.get(item, 0) + qty

                covered_aisles = multi_greedy_aisle_select(tentative_demand, aisles)
                if not covered_aisles:
                    continue

                ratio = tentative_units / len(covered_aisles)
                if ratio > best_candidate_ratio:
                    best_candidate_ratio = ratio
                    best_candidate = candidate
                    best_candidate_aisles = covered_aisles

            if best_candidate == -1:
                break

            selected_orders.append(best_candidate)
            remaining_candidates.remove(best_candidate)
            total_units += order_units[best_candidate]
            for item, qty in orders[best_candidate].items():
                current_demand[item] = current_demand.get(item, 0) + qty

            if total_units >= lb and best_candidate_ratio > best_obj:
                best_obj = best_candidate_ratio
                best_orders = list(selected_orders)
                best_aisles = best_candidate_aisles

        if not best_orders or not best_aisles:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        total_items = sum(order_units[o] for o in best_orders)
        objective = total_items / len(best_aisles)
        return {
            "selected_orders": best_orders,
            "visited_aisles": best_aisles,
            "objective": objective,
        }
