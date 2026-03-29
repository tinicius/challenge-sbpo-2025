from mealpy.evolutionary_based.GA import BaseGA

from impl.utils.mealpy_binary_solver import MealpyBinaryMetaheuristicSolver


class GeneticAlgorithmHeuristic(MealpyBinaryMetaheuristicSolver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)

        self.population_size = max(10, int(self.config.get("population_size", 80)))
        self.generations = max(1, int(self.config.get("generations", 120)))
        self.crossover_rate = float(self.config.get("crossover_rate", 0.85))
        self.tournament_size = max(2, int(self.config.get("tournament_size", 3)))
        self.penalty_scale = float(self.config.get("penalty_scale", 10.0))
        self.penalty_lb = float(self.config.get("penalty_lb", 1.0))
        self.penalty_ub = float(self.config.get("penalty_ub", 1.0))
        self.penalty_stock = float(self.config.get("penalty_stock", 2.0))

    def solve(self) -> tuple[list[int], list[int]]:
        configured_mutation = self.config.get("mutation_rate")
        if configured_mutation is None:
            dynamic = 1.0 / max(1, self.n_orders)
            mutation_rate = max(0.005, min(0.1, dynamic))
        else:
            mutation_rate = float(configured_mutation)

        model = BaseGA(
            epoch=self.generations,
            pop_size=self.population_size,
            pc=self.crossover_rate,
            pm=mutation_rate,
            selection="tournament",
            k_way=max(0.01, min(1.0, self.tournament_size / max(1, self.population_size))),
            crossover="uniform",
            mutation="flip",
            mutation_multipoints=True,
        )
        return self._solve_with_mealpy_model(model)
