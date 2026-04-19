from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class GreedyRatioContribution(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "greedy_ratio_contribution"

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        orders, aisles = inst.orders, inst.aisles
        lb, ub = inst.lb, inst.ub
        order_units = [sum(o.values()) for o in orders]

        selected_orders: list[int] = []
        current_demand: dict[int, int] = {}
        total_units = 0
        current_aisles: list[int] = []

        best_obj = -1.0
        best_orders: list[int] = []
        best_aisles: list[int] = []

        remaining = set(range(inst.nOrders))

        while remaining:
            A = len(current_aisles)
            current_ratio = total_units / A if A > 0 else 0.0

            best_candidate = -1
            best_delta = -float("inf")
            best_cand_aisles: list[int] = []

            for o in remaining:
                if total_units + order_units[o] > ub:
                    continue

                tentative_demand = dict(current_demand)
                for item, qty in orders[o].items():
                    tentative_demand[item] = tentative_demand.get(item, 0) + qty

                t_aisles = multi_greedy_aisle_select(tentative_demand, aisles)
                if not t_aisles:
                    continue

                delta = (total_units + order_units[o]) / len(t_aisles) - current_ratio
                if delta > best_delta:
                    best_delta = delta
                    best_candidate = o
                    best_cand_aisles = t_aisles

            if best_candidate == -1:
                break

            selected_orders.append(best_candidate)
            remaining.remove(best_candidate)
            total_units += order_units[best_candidate]
            for item, qty in orders[best_candidate].items():
                current_demand[item] = current_demand.get(item, 0) + qty
            current_aisles = best_cand_aisles

            obj = total_units / len(current_aisles)
            if total_units >= lb and obj > best_obj:
                best_obj = obj
                best_orders = list(selected_orders)
                best_aisles = list(current_aisles)

        if not best_orders:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        return {
            "selected_orders": best_orders,
            "visited_aisles": best_aisles,
            "objective": sum(order_units[o] for o in best_orders) / len(best_aisles),
        }
