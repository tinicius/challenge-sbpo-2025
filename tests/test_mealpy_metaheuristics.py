import unittest

from impl.ant_colony_optimization import AntColonyOptimizationHeuristic
from impl.genetic_algorithm_heuristic import GeneticAlgorithmHeuristic
from impl.simulated_annealing_heuristic import SimulatedAnnealingHeuristic
from impl.tabu_search_heuristic import TabuSearchHeuristic
from models.solver import ProblemInput
from utils.wave_order_picking import WaveOrderPicking


class MealpyMetaheuristicsTest(unittest.TestCase):
    def setUp(self):
        orders = [
            {0: 3, 2: 1},
            {1: 1, 3: 1},
            {2: 1, 4: 2},
            {0: 1, 2: 2, 3: 1, 4: 1},
            {1: 1},
        ]
        aisles = [
            {0: 2, 1: 1, 2: 1, 4: 1},
            {0: 2, 1: 1, 2: 2, 4: 1},
            {1: 2, 3: 1, 4: 2},
            {0: 2, 1: 1, 3: 1, 4: 1},
            {1: 1, 2: 2, 3: 1, 4: 2},
        ]
        self.problem_input = ProblemInput(
            nOrders=5,
            nItems=5,
            nAisles=5,
            orders=orders,
            aisles=aisles,
            lb=5,
            ub=12,
        )
        self.checker = WaveOrderPicking()
        self.checker.load_problem_input(self.problem_input)

    def _assert_solver_returns_feasible_nonempty_solution(self, solver_cls, config):
        solver = solver_cls(self.problem_input, config)
        selected_orders, visited_aisles = solver.solve()

        self.assertTrue(selected_orders)
        self.assertTrue(visited_aisles)
        self.assertTrue(
            self.checker.is_solution_feasible(selected_orders, visited_aisles),
            msg=f"Invalid solution for {solver_cls.__name__}: "
            f"orders={selected_orders}, aisles={visited_aisles}",
        )
        self.assertGreater(
            self.checker.compute_objective_function(selected_orders, visited_aisles), 0.0
        )

    def test_genetic_algorithm_mealpy(self):
        self._assert_solver_returns_feasible_nonempty_solution(
            GeneticAlgorithmHeuristic,
            {"seed": 1, "population_size": 40, "generations": 60},
        )

    def test_simulated_annealing_mealpy(self):
        self._assert_solver_returns_feasible_nonempty_solution(
            SimulatedAnnealingHeuristic,
            {"seed": 1, "max_iterations": 300, "initial_temp": 5.0},
        )

    def test_tabu_search_mealpy(self):
        self._assert_solver_returns_feasible_nonempty_solution(
            TabuSearchHeuristic,
            {"seed": 1, "max_iterations": 120, "neighborhood_samples": 20},
        )

    def test_ant_colony_mealpy(self):
        self._assert_solver_returns_feasible_nonempty_solution(
            AntColonyOptimizationHeuristic,
            {"seed": 1, "n_ants": 20, "n_iterations": 40, "candidate_list_size": 10},
        )


if __name__ == "__main__":
    unittest.main()
