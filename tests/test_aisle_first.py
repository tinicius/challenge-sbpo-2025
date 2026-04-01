import unittest

from algorithms.aisle_first import AisleFirstHeuristic
from problems.base import ProblemInput
from problems.validation import is_solution_feasible


class TestAisleFirstHeuristic(unittest.TestCase):

    def test_finds_expected_solution_on_reference_instance(self):
        instance = ProblemInput(
            nOrders=5,
            nItems=5,
            nAisles=5,
            orders=[
                {0: 3, 2: 1},
                {1: 1, 3: 1},
                {2: 1, 4: 2},
                {0: 1, 2: 2, 3: 1, 4: 1},
                {1: 1},
            ],
            aisles=[
                {0: 2, 1: 1, 2: 1, 4: 1},
                {0: 2, 1: 1, 2: 2, 4: 1},
                {1: 2, 3: 1, 4: 2},
                {0: 2, 1: 1, 3: 1, 4: 1},
                {1: 1, 2: 2, 3: 1, 4: 2},
            ],
            lb=5,
            ub=12,
        )

        result = AisleFirstHeuristic(params={}).solve(instance)

        self.assertTrue(
            is_solution_feasible(
                instance, result["selected_orders"], result["visited_aisles"]
            )
        )
        self.assertEqual(5.0, result["objective"])


if __name__ == "__main__":
    unittest.main()
