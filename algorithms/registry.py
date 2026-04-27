"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.aisle_first.aisle_first_exact_orders import AisleFirstExactOrders
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.ga.ga_heuristic import GeneticAlgorithm
from algorithms.pan_liu.pan_liu_heuristic import PanLiuHeuristic
from algorithms.seed.seed_heuristic import SeedHeuristic
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.dinkelbach_mip import DinkelbachMIP
from algorithms.dinkelbach_alns.algorithm import DinkelbachALNS


REGISTRY = {
    # Order based heuristics
    "simple": SimpleHeuristic,
    "seed": SeedHeuristic,
    # Genetic
    "ga": GeneticAlgorithm,
    "pan_liu": PanLiuHeuristic,
    # Aisle based heuristics
    "aisle_first": AisleFirstHeuristic,
    "aisle_first_exact": AisleFirstExactOrders,
    # MIP based heuristics
    "dinkelbach_mip": DinkelbachMIP,
    # Matheuristic
    "dalns": DinkelbachALNS,
}
