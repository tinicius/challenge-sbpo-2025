from mealpy.physics_based.SA import OriginalSA

from impl.utils.mealpy_binary_solver import MealpyBinaryMetaheuristicSolver


class SimulatedAnnealingHeuristic(MealpyBinaryMetaheuristicSolver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)

        self.max_iterations = max(1, int(self.config.get("max_iterations", 4000)))
        self.initial_temp = max(1e-9, float(self.config.get("initial_temp", 2.5)))
        self.penalty_scale = float(self.config.get("penalty_scale", 10.0))
        self.penalty_lb = float(self.config.get("penalty_lb", 1.0))
        self.penalty_ub = float(self.config.get("penalty_ub", 1.0))
        self.penalty_stock = float(self.config.get("penalty_stock", 2.0))

    def solve(self) -> tuple[list[int], list[int]]:
        model = OriginalSA(
            epoch=self.max_iterations,
            pop_size=2,
            temp_init=self.initial_temp,
            step_size=float(self.config.get("step_size", 0.1)),
        )
        return self._solve_with_mealpy_model(model)
