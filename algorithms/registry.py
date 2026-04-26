"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.ga.ga_heuristic import GeneticAlgorithm
from algorithms.ga.ga_heuristic_full import GeneticAlgorithmFull
from algorithms.pan_liu.pan_liu_heuristic import PanLiuHeuristic
from algorithms.seed.seed_heuristic import SeedHeuristic
from algorithms.simple.simple_cpp_heuristic import SimpleCppHeuristic
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.dinkelbach_mip import DinkelbachMIP


REGISTRY = {
    # Order based heuristics
    "simple": SimpleHeuristic,
    "simple_cpp": SimpleCppHeuristic,
    "seed": SeedHeuristic,
    # Genetic
    "ga": GeneticAlgorithm,
    "ga_full": GeneticAlgorithmFull,
    "pan_liu": PanLiuHeuristic,
    # Aisle based heuristics
    "aisle_first": AisleFirstHeuristic,
    # MIP based heuristics
    "dinkelbach_mip": DinkelbachMIP,
}
