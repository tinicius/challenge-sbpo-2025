from mealpy import TransferBinaryVar

from impl.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from models.solver import Solver


class MealpyBinaryMetaheuristicSolver(Solver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)
        self.order_units = [sum(order.values()) for order in self.orders]

        self.penalty_scale = float(self.config.get("penalty_scale", 10.0))
        self.penalty_lb = float(self.config.get("penalty_lb", 1.0))
        self.penalty_ub = float(self.config.get("penalty_ub", 1.0))
        self.penalty_stock = float(self.config.get("penalty_stock", 2.0))

    def _decode_binary(self, solution) -> list[int]:
        return [1 if float(value) >= 0.5 else 0 for value in solution]

    def _build_demand(self, chromosome: list[int]) -> tuple[list[int], dict[int, int], int]:
        selected_orders: list[int] = []
        demand: dict[int, int] = {}
        units = 0

        for order_idx, bit in enumerate(chromosome):
            if bit == 0:
                continue
            selected_orders.append(order_idx)
            units += self.order_units[order_idx]
            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        return selected_orders, demand, units

    def _compute_supply(self, aisle_indices: list[int]) -> dict[int, int]:
        supply: dict[int, int] = {}
        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                supply[item] = supply.get(item, 0) + qty
        return supply

    def _evaluate_chromosome(self, chromosome: list[int]) -> dict:
        selected_orders, demand, units = self._build_demand(chromosome)

        if not selected_orders:
            return {
                "fitness": -self.penalty_scale,
                "objective": 0.0,
                "units": 0,
                "selected_orders": [],
                "visited_aisles": [],
                "feasible": False,
            }

        visited_aisles = multi_greedy_aisle_select(demand, self.aisles)
        supply = self._compute_supply(visited_aisles)

        stock_deficit = 0
        for item, req_qty in demand.items():
            stock_deficit += max(0, req_qty - supply.get(item, 0))

        lb_violation = max(0, self.lb - units)
        ub_violation = max(0, units - self.ub)

        raw_objective = units / len(visited_aisles) if visited_aisles else 0.0
        normalized_lb = lb_violation / max(1, self.lb)
        normalized_ub = ub_violation / max(1, self.ub)
        normalized_stock = stock_deficit / max(1, sum(demand.values()))

        penalty = (
            self.penalty_lb * normalized_lb
            + self.penalty_ub * normalized_ub
            + self.penalty_stock * normalized_stock
        )
        fitness = raw_objective - (self.penalty_scale * penalty)

        feasible = (
            self.lb <= units <= self.ub and stock_deficit == 0 and len(visited_aisles) > 0
        )

        return {
            "fitness": fitness,
            "objective": raw_objective,
            "units": units,
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "feasible": feasible,
        }

    def _objective(self, solution) -> float:
        chromosome = self._decode_binary(solution)
        return self._evaluate_chromosome(chromosome)["fitness"]

    def _best_feasible_from_solutions(self, solutions) -> dict | None:
        best = None
        cache: dict[tuple[int, ...], dict] = {}

        for solution in solutions:
            chromosome = self._decode_binary(solution)
            key = tuple(chromosome)
            evaluated = cache.get(key)
            if evaluated is None:
                evaluated = self._evaluate_chromosome(chromosome)
                cache[key] = evaluated

            if not evaluated["feasible"]:
                continue

            if best is None:
                best = evaluated
                continue

            if evaluated["objective"] > best["objective"]:
                best = evaluated
            elif evaluated["objective"] == best["objective"]:
                if len(evaluated["visited_aisles"]) < len(best["visited_aisles"]):
                    best = evaluated
                elif (
                    len(evaluated["visited_aisles"]) == len(best["visited_aisles"])
                    and evaluated["units"] > best["units"]
                ):
                    best = evaluated

        return best

    def _solve_with_mealpy_model(self, model) -> tuple[list[int], list[int]]:
        if self.n_orders == 0 or self.n_aisles == 0:
            return [], []

        seed = self.config.get("random_seed", self.config.get("seed"))
        problem = {
            "bounds": TransferBinaryVar(n_vars=self.n_orders, name="orders"),
            "minmax": "max",
            "obj_func": self._objective,
            "log_to": "None",
        }
        if seed is None:
            best_agent = model.solve(problem)
        else:
            best_agent = model.solve(problem, seed=seed)

        candidate_solutions = [best_agent.solution]
        if hasattr(model, "pop") and model.pop is not None:
            candidate_solutions.extend(agent.solution for agent in model.pop)

        best = self._best_feasible_from_solutions(candidate_solutions)
        if best is None:
            return [], []

        selected_orders = sorted(set(best["selected_orders"]))
        visited_aisles = sorted(set(best["visited_aisles"]))
        return selected_orders, visited_aisles
