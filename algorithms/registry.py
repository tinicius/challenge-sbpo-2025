"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.simple_heuristic import SimpleHeuristic
from algorithms.similarity_heuristic import SimilarityHeuristic
from algorithms.greedy_order_similarity_aggregation import GreedyOrderSimilarityAggregationHeuristic
from algorithms.max_density_order_insertion import MaxDensityOrderInsertionHeuristic
from algorithms.greedy_aisle_density import GreedyAisleDensityHeuristic
from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.aisle_cluster_expansion import AisleClusterExpansion
from algorithms.ping_pong_alternating import PingPongAlternatingHeuristic
from algorithms.local_search import LocalSearchHeuristic
from algorithms.simulated_annealing import SimulatedAnnealingHeuristic as SABasic
from algorithms.simulated_annealing_heuristic import SimulatedAnnealingHeuristic as SAAdvanced
from algorithms.genetic_algorithm_heuristic import GeneticAlgorithmHeuristic
from algorithms.ant_colony_optimization import AntColonyOptimizationHeuristic
from algorithms.tabu_search_heuristic import TabuSearchHeuristic
from algorithms.grasp_heuristic import GRASPHeuristic
from algorithms.dinkelbach_milp import DinkelbachMILPSolver
from algorithms.dinkelbach_local_search import DinkelbachLocalSearchHeuristic

REGISTRY = {
    # Greedy heuristics
    "simple": SimpleHeuristic,
    "simple_multi": SimpleHeuristic,
    "similar": SimilarityHeuristic,
    "similar_multi": SimilarityHeuristic,
    "diff": SimilarityHeuristic,
    "diff_multi": SimilarityHeuristic,
    "similarity_aggregation": GreedyOrderSimilarityAggregationHeuristic,
    "max_density_insert": MaxDensityOrderInsertionHeuristic,
    "greedy_aisle_density": GreedyAisleDensityHeuristic,
    # Construction heuristics
    "aisle_first": AisleFirstHeuristic,
    "aisle_cluster_expansion": AisleClusterExpansion,
    "ping_pong": PingPongAlternatingHeuristic,
    # Metaheuristics
    "local_search": LocalSearchHeuristic,
    "sa_basic": SABasic,
    "sa_advanced": SAAdvanced,
    "genetic": GeneticAlgorithmHeuristic,
    "aco": AntColonyOptimizationHeuristic,
    "tabu_search": TabuSearchHeuristic,
    "grasp": GRASPHeuristic,
    # Exact / Hybrid
    "dinkelbach_milp": DinkelbachMILPSolver,
    "dinkelbach_local_search": DinkelbachLocalSearchHeuristic,
}
