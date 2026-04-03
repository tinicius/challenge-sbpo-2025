from algorithms.aisle_first import AisleFirstHeuristic
from problems.base import ProblemInput, load_instance

instance_path = "datasets/a/instance_0001.txt"

instance = load_instance(instance_path)

solver = AisleFirstHeuristic(params={"max_k": 1, "order_strategy": "size_desc"})

result = solver.solve(instance)

print(result)
