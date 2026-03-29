from mealpy.math_based.TS import OriginalTS

from impl.utils.mealpy_binary_solver import MealpyBinaryMetaheuristicSolver


class RobustOriginalTS(OriginalTS):
    def evolve(self, epoch):
        candidates = self.generator.normal(
            loc=self.x,
            scale=self.perturbation_scale,
            size=(self.neighbour_size, self.problem.n_dims),
        )
        list_candidates = []
        for candidate in candidates:
            pos_new = self.correct_solution(candidate)
            if tuple(pos_new) in self.tabu_list:
                continue
            agent = self.generate_empty_agent(pos_new)
            list_candidates.append(agent)
            if self.mode not in self.AVAILABLE_MODES:
                list_candidates[-1].target = self.get_target(pos_new)

        if not list_candidates:
            random_agent = self.generate_agent()
            self.x = random_agent.solution
            self.tabu_list.append(tuple(self.x))
            self.pop.append(random_agent)
            if len(self.tabu_list) > self.tabu_size:
                self.tabu_list.pop(0)
                self.pop.pop(0)
            return

        list_candidates = self.update_target_for_population(list_candidates)
        best_candidate = self.get_best_agent(list_candidates, self.problem.minmax)
        self.x = best_candidate.solution
        self.tabu_list.append(tuple(self.x))
        self.pop.append(best_candidate)
        if len(self.tabu_list) > self.tabu_size:
            self.tabu_list.pop(0)
            self.pop.pop(0)


class TabuSearchHeuristic(MealpyBinaryMetaheuristicSolver):
    def __init__(self, input, config: dict = {}):
        super().__init__(input, config)
        self.max_iterations = max(1, int(self.config.get("max_iterations", 180)))
        self.penalty_scale = float(self.config.get("penalty_scale", 10.0))
        self.penalty_lb = float(self.config.get("penalty_lb", 1.0))
        self.penalty_ub = float(self.config.get("penalty_ub", 1.0))
        self.penalty_stock = float(self.config.get("penalty_stock", 2.0))

    def solve(self) -> tuple[list[int], list[int]]:
        model = RobustOriginalTS(
            epoch=self.max_iterations,
            pop_size=2,
            tabu_size=max(2, int(self.config.get("tabu_tenure_orders", 10))),
            neighbour_size=max(2, int(self.config.get("neighborhood_samples", 40))),
            perturbation_scale=float(self.config.get("perturbation_scale", 0.05)),
        )
        return self._solve_with_mealpy_model(model)
