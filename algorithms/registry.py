"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.simple_heuristic import SimpleHeuristic
from algorithms.similarity_heuristic import SimilarityHeuristic
from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.aisle_cluster_expansion import AisleClusterExpansion
from algorithms.greedy_set_cover import GreedySetCover
from algorithms.greedy_ratio_contribution import GreedyRatioContribution
from algorithms.knapsack_density import KnapsackDensitySolver
from algorithms.knapsack_dp_relaxation import KnapsackDPRelaxation
from algorithms.knapsack_seed import KnapsackSeedSolver
from algorithms.knapsack_solver import KnapsackSolver
from algorithms.lp_aisle_focus import LPAisleFocus
from algorithms.lagrangian_relaxation import LagrangianRelaxation
from algorithms.column_generation import ColumnGeneration
from algorithms.min_aisle_cover import MinAisleCover
from algorithms.order_aisle_similarity import OrderAisleSimilarity
from algorithms.order_overlap_cluster import OrderOverlapCluster
from algorithms.exact_mip import ExactMIP
from algorithms.dinkelbach_mip import DinkelbachMIP


REGISTRY = {
    # Greedy heuristics
    "simple": SimpleHeuristic,
    "similar": SimilarityHeuristic,
    "aisle_first": AisleFirstHeuristic,
    "order_aisle_similarity": OrderAisleSimilarity,
    "order_overlap_cluster": OrderOverlapCluster,
    "aisle_cluster_expansion": AisleClusterExpansion,
    "greedy_set_cover": GreedySetCover,
    "greedy_ratio_contribution": GreedyRatioContribution,
    # Knapsack-based
    "knapsack": KnapsackSolver,
    "knapsack_seed": KnapsackSeedSolver,
    "knapsack_density": KnapsackDensitySolver,
    "knapsack_dp_relaxation": KnapsackDPRelaxation,
    # LP-based
    "lp_aisle_focus": LPAisleFocus,
    "column_generation": ColumnGeneration,
    # Set-cover
    "min_aisle_cover": MinAisleCover,
    # Lagrangian
    "lagrangian_relaxation": LagrangianRelaxation,
    # Exact
    "exact_mip": ExactMIP,
    "dinkelbach_mip": DinkelbachMIP,
}
