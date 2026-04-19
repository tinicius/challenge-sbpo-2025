"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.dinkelbach_mip import DinkelbachMIP


REGISTRY = {
    # Order based heuristics
    "simple": SimpleHeuristic,
    # Aisle based heuristics
    #
    # MIP based heuristics
    "dinkelbach_mip": DinkelbachMIP,
}
