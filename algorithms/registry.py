"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.similarity_heuristic import SimilarityHeuristic
from algorithms.dinkelbach_mip import DinkelbachMIP


REGISTRY = {
    # Order based heuristics
    "simple": SimpleHeuristic,
    "similar": SimilarityHeuristic,
    # Aisle based heuristics
    #
    # MIP based heuristics
    "dinkelbach_mip": DinkelbachMIP,
}
