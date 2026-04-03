from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.order_aisle_similarity import OrderAisleSimilarity
from problems.base import ProblemInput, load_instance

instance_path = "datasets/a/instance_0007.txt"

instance = load_instance(instance_path)

solver = OrderAisleSimilarity(params={})

result = solver.solve(instance)

print(result)
