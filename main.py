from algorithms.seed.seed_ilp_gurobi_heuristic import SeedIlpGurobiHeuristic
from problems.base import ProblemInput, load_instance

instance_path = "datasets/a/instance_0004.txt"

instance = load_instance(instance_path)

solver = SeedIlpGurobiHeuristic(
    {
        "seed_strategy": "least_distinct_items",
        "gurobi_time_limit": "30",
        "gurobi_threads": "0",
    }
)


result = solver.solve(instance)

print(result)
