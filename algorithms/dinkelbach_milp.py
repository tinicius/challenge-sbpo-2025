import time
from collections import defaultdict

from ortools.linear_solver import pywraplp

from algorithms.base import Algorithm
from problems.base import ProblemInput


class DinkelbachMILPSolver(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "dinkelbach_milp"

    def _solve_subproblem(
        self,
        lambda_: float,
        time_limit: float,
        units_per_order: list[int],
        items_in_orders: dict[int, list[tuple[int, int]]],
        items_in_aisles: dict[int, list[tuple[int, int]]],
        relevant_items: set[int],
    ):
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            return None, None, None

        solver.SetTimeLimit(int(time_limit * 1000))

        x = [solver.BoolVar(f"x_{o}") for o in range(self.n_orders)]
        y = [solver.BoolVar(f"y_{a}") for a in range(self.n_aisles)]

        # Objective: max sum(x_o * units_o) - lambda * sum(y_a)
        objective = solver.Objective()
        for o in range(self.n_orders):
            objective.SetCoefficient(x[o], units_per_order[o])
        for a in range(self.n_aisles):
            objective.SetCoefficient(y[a], -lambda_)
        objective.SetMaximization()

        # Wave size bounds: LB <= sum(x_o * units_o) <= UB
        wave_ct = solver.Constraint(self.lb, self.ub, "wave_size")
        for o in range(self.n_orders):
            wave_ct.SetCoefficient(x[o], units_per_order[o])

        # Stock constraints: for each item i,
        # sum(x_o * demand(o,i)) <= sum(y_a * stock(a,i))
        for item in relevant_items:
            ct = solver.Constraint(-solver.infinity(), 0, f"stock_{item}")
            for o, qty in items_in_orders[item]:
                ct.SetCoefficient(x[o], qty)
            for a, qty in items_in_aisles[item]:
                ct.SetCoefficient(y[a], -qty)

        status = solver.Solve()

        x_sol = [x[o].solution_value() for o in range(self.n_orders)]
        y_sol = [y[a].solution_value() for a in range(self.n_aisles)]

        return x_sol, y_sol, status

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        max_iterations = self.config.get("max_iterations", 20)
        epsilon = self.config.get("epsilon", 1e-6)
        time_limit_per_sub = self.config.get("time_limit", 2.0)
        total_time_limit = self.config.get("total_time_limit", 4.5)

        # Precompute
        units_per_order = []
        for o in range(self.n_orders):
            units_per_order.append(sum(self.orders[o].values()))

        items_in_orders: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for o in range(self.n_orders):
            for item, qty in self.orders[o].items():
                items_in_orders[item].append((o, qty))

        items_in_aisles: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for a in range(self.n_aisles):
            for item, qty in self.aisles[a].items():
                items_in_aisles[item].append((a, qty))

        relevant_items = set(items_in_orders.keys())

        # Dinkelbach loop
        lambda_ = 0.0
        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_obj = -1.0
        start_time = time.perf_counter()

        for iteration in range(max_iterations):
            remaining = total_time_limit - (time.perf_counter() - start_time)
            if remaining <= 0.1:
                break
            sub_limit = min(time_limit_per_sub, remaining - 0.05)

            x_sol, y_sol, status = self._solve_subproblem(
                lambda_,
                sub_limit,
                units_per_order,
                items_in_orders,
                items_in_aisles,
                relevant_items,
            )

            if status is None:
                break
            if status not in (
                pywraplp.Solver.OPTIMAL,
                pywraplp.Solver.FEASIBLE,
            ):
                break

            selected_orders = [o for o in range(self.n_orders) if x_sol[o] > 0.5]
            visited_aisles = [a for a in range(self.n_aisles) if y_sol[a] > 0.5]

            n_val = sum(units_per_order[o] for o in selected_orders)
            d_val = len(visited_aisles)

            if d_val == 0:
                break

            current_obj = n_val / d_val
            if current_obj > best_obj:
                best_obj = current_obj
                best_orders = selected_orders
                best_aisles = visited_aisles

            f_lambda = n_val - lambda_ * d_val
            if abs(f_lambda) <= epsilon:
                break

            lambda_ = current_obj

        selected_orders = best_orders
        visited_aisles = best_aisles

        if not selected_orders or not visited_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}
