"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.grasp.grasp_heuristic import GraspHeuristic
from algorithms.sa.sa_heuristic import SimulatedAnnealing
from algorithms.seed.seed_heuristic import SeedHeuristic
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.dinkelbach_mip import DinkelbachMIP


REGISTRY = {
    # Order based heuristics
    "simple": SimpleHeuristic,
    "seed": SeedHeuristic,
    "grasp": GraspHeuristic,
    "sa": SimulatedAnnealing,
    # Aisle based heuristics
    "aisle_first": AisleFirstHeuristic,
    # MIP based heuristics
    "dinkelbach_mip": DinkelbachMIP,
}
