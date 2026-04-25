"""Algorithm registry: maps config names to Algorithm classes."""

from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.aisle_first.aisle_grasp_cpp_heuristic import AisleGraspCppHeuristic
from algorithms.aisle_first.aisle_grasp_heuristic import AisleGraspHeuristic
from algorithms.ga.ga_heuristic import GeneticAlgorithm
from algorithms.grasp.grasp_heuristic import GraspHeuristic
from algorithms.grasp.grasp_cpp_heuristic import GraspCppHeuristic
from algorithms.pan_liu.pan_liu_heuristic import PanLiuHeuristic
from algorithms.sa.sa_heuristic import SimulatedAnnealing
from algorithms.seed.seed_ilp_gurobi_heuristic import SeedIlpGurobiHeuristic
from algorithms.seed.seed_heuristic import SeedHeuristic
from algorithms.simple.simple_cpp_heuristic import SimpleCppHeuristic
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.dinkelbach_mip import DinkelbachMIP


REGISTRY = {
    # Order based heuristics
    "simple": SimpleHeuristic,
    "simple_cpp": SimpleCppHeuristic,
    "seed": SeedHeuristic,
    "seed_ilp_gurobi": SeedIlpGurobiHeuristic,
    "grasp": GraspHeuristic,
    "grasp_cpp": GraspCppHeuristic,
    "sa": SimulatedAnnealing,
    "ga": GeneticAlgorithm,
    "pan_liu": PanLiuHeuristic,
    # Aisle based heuristics
    "aisle_first": AisleFirstHeuristic,
    "aisle_grasp": AisleGraspHeuristic,
    "aisle_grasp_cpp": AisleGraspCppHeuristic,
    # MIP based heuristics
    "dinkelbach_mip": DinkelbachMIP,
}
