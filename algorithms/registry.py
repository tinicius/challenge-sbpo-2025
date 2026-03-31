"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.simple_heuristic import SimpleHeuristic
from algorithms.similarity_heuristic import SimilarityHeuristic
from algorithms.aisle_first import AisleFirstHeuristic
from algorithms.aisle_cluster_expansion import AisleClusterExpansion


REGISTRY = {
    # Greedy heuristics
    "simple": SimpleHeuristic,
    "simple_multi": SimpleHeuristic,
    "similar": SimilarityHeuristic,
    "similar_multi": SimilarityHeuristic,
    "diff": SimilarityHeuristic,
    "diff_multi": SimilarityHeuristic,
    "aisle_first": AisleFirstHeuristic,
    "aisle_cluster_expansion": AisleClusterExpansion,
}
