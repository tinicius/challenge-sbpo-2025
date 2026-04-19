from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.order_aisle_similarity import OrderAisleSimilarity
from algorithms.order_overlap_cluster import OrderOverlapCluster
from algorithms.simple_heuristic import SimpleHeuristic
from algorithms.exact_mip import ExactMIP
from problems.base import ProblemInput, load_instance

instance_path = "datasets/a/instance_0013.txt"
# 117.38461538461539

instance = load_instance(instance_path)

solver = AisleFirstHeuristic(params={"max_seeds": 1, "order_strategy": "size_desc"})

# time_limit: 590
#       max_dinkelbach_iters: 30
#       num_workers: 8

result = solver.solve(instance)

print(result)
