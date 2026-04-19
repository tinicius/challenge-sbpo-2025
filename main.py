from algorithms.order_aisle_similarity import OrderAisleSimilarity
from algorithms.simple.simple_heuristic import SimpleHeuristic
from problems.base import ProblemInput, load_instance

instance_path = "datasets/a/instance_0013.txt"

instance = load_instance(instance_path)

solver = SimpleHeuristic(
    {
        "order": "desc",
        "greedy": "multi",
    }
)


result = solver.solve(instance)

print(result)
