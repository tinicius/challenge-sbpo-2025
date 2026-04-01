"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.simple_heuristic import SimpleHeuristic
from algorithms.similarity_heuristic import SimilarityHeuristic
from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.aisle_cluster_expansion import AisleClusterExpansion
from algorithms.greedy_set_cover import GreedySetCover
from algorithms.lp_aisle_focus import LPAisleFocus
from algorithms.lagrangian_relaxation import LagrangianRelaxation


REGISTRY = {
    # Greedy heuristics
    "simple": SimpleHeuristic,
    "similar": SimilarityHeuristic,
    "aisle_first": AisleFirstHeuristic,
    "aisle_cluster_expansion": AisleClusterExpansion,
    "greedy_set_cover": GreedySetCover,
    # LP-based
    "lp_aisle_focus": LPAisleFocus,
    # Lagrangian
    "lagrangian_relaxation": LagrangianRelaxation,
}
