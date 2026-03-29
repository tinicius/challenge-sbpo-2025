from mealpy.swarm_based.ACOR import OriginalACOR

from impl.utils.mealpy_binary_solver import MealpyBinaryMetaheuristicSolver


class AntColonyOptimizationHeuristic(MealpyBinaryMetaheuristicSolver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)
        self.n_ants = max(2, int(self.config.get("n_ants", 30)))
        self.n_iterations = max(1, int(self.config.get("n_iterations", 60)))
        self.penalty_scale = float(self.config.get("penalty_scale", 10.0))
        self.penalty_lb = float(self.config.get("penalty_lb", 1.0))
        self.penalty_ub = float(self.config.get("penalty_ub", 1.0))
        self.penalty_stock = float(self.config.get("penalty_stock", 2.0))

    def solve(self) -> tuple[list[int], list[int]]:
        model = OriginalACOR(
            epoch=self.n_iterations,
            pop_size=self.n_ants,
            sample_count=max(2, int(self.config.get("candidate_list_size", 25))),
            intent_factor=float(self.config.get("intent_factor", 0.5)),
            zeta=float(self.config.get("zeta", 1.0)),
        )
        return self._solve_with_mealpy_model(model)
