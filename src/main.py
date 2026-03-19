from typing import Callable
from impl.simple_heuristic import SimpleHeuristic
from impl.similarity_heuristic import SimilarityHeuristic
from impl.order_batching_heuristics import (
    LargestFirstHeuristic,
    GreedyRatioHeuristic,
    LocalSearchHeuristic,
    GRASPHeuristic,
)

import time

import random


from models.solver import Solver, ProblemInput
import os
from sys import argv

from utils.generate_output import generate_output
from utils.read_input import read_input
from utils.wave_order_picking import WaveOrderPicking

import csv


class RunConfig:
    def __init__(
        self,
        name: str,
        solver_class: type[Solver],
        get_runs: Callable[[ProblemInput], list[dict]],
    ):
        self.name = name
        self.solver_class = solver_class
        self.get_runs = get_runs


def solve(output_file, solver):
    solution = solver.solve()

    generate_output(
        output_file, len(solution[0]), len(solution[1]), solution[0], solution[1]
    )


def process(solver_config: RunConfig, input_folder: str, output_folder: str):

    objectives_dir = os.path.join(os.path.dirname(__file__), "..", "objectives_output")
    os.makedirs(objectives_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.normpath(input_folder))
    results = []

    for filename in sorted(os.listdir(input_folder)):

        if not filename.endswith(".txt"):
            continue

        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)

        wave_order_picking = WaveOrderPicking()
        wave_order_picking.read_input(input_file)

        input = read_input(input_file)

        objectives_sum = 0
        number_feasibles_solutions = 0

        run = 0

        start = time.perf_counter()

        for run_config in solver_config.get_runs(input):

            total_runs = len(solver_config.get_runs(input))

            solver = solver_config.solver_class(input, run_config)

            solve(output_file, solver)

            end = time.perf_counter()

            selected_orders, visited_aisles = wave_order_picking.read_output(
                output_file
            )

            is_feasible = wave_order_picking.is_solution_feasible(
                selected_orders, visited_aisles
            )

            objective_value = wave_order_picking.compute_objective_function(
                selected_orders, visited_aisles
            )

            if is_feasible:
                objectives_sum += objective_value
                number_feasibles_solutions += 1

            print(
                f"{solver_config.name} - {filename} - {run + 1}/{total_runs} - Time: {end - start:.2f}s - Objective: {objective_value} - Feasible: {is_feasible}",
                flush=True,
            )

            run += 1

        avg = (
            objectives_sum / number_feasibles_solutions
            if number_feasibles_solutions > 0
            else -1
        )

        results.append((dataset_name, filename, avg))

    csv_path = os.path.join(
        objectives_dir,
        f"{solver_config.name}_{dataset_name}_{solver_config.name}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "instance", "best_objective"])
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")


runs = 5


def _random_seeds(n_orders: int, n: int = runs) -> list[list[int]]:
    """Return *n* independently shuffled permutations of order indices."""
    base = list(range(n_orders))
    return [random.sample(base, len(base)) for _ in range(n)]


def simple_heuristic_runs(input: ProblemInput) -> list[dict]:

    configs = []

    for seed in _random_seeds(input.nOrders):
        configs.append({"seed": seed})

    return configs


def similar_heuristic_runs(input: ProblemInput) -> list[dict]:

    configs = []

    for seed in _random_seeds(input.nOrders):
        configs.append({"seed": seed, "reverse": True})

    return configs


def diff_heuristic_runs(input: ProblemInput) -> list[dict]:

    configs = []

    for seed in _random_seeds(input.nOrders):
        configs.append({"seed": seed, "reverse": False})

    return configs


def largest_first_runs(input: ProblemInput) -> list[dict]:

    return [{"seed": seed} for seed in _random_seeds(input.nOrders)]


def greedy_ratio_runs(input: ProblemInput) -> list[dict]:

    return [{"seed": seed} for seed in _random_seeds(input.nOrders)]


def local_search_runs(input: ProblemInput) -> list[dict]:

    return [
        {"seed": seed, "construction": "largest", "max_iter": 100}
        for seed in _random_seeds(input.nOrders)
    ]


def grasp_runs(input: ProblemInput) -> list[dict]:
    # Five runs with varying alpha values to balance greediness and diversity
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    return [{"n_iter": 5, "alpha": a, "max_ls_iter": 30} for a in alphas]


if __name__ == "__main__":

    input_folder = "datasets/a"
    output_folder = "output"

    solver_configs = [
        RunConfig(
            "simple",
            SimpleHeuristic,
            simple_heuristic_runs,
        ),
        RunConfig(
            "similar",
            SimilarityHeuristic,
            similar_heuristic_runs,
        ),
        RunConfig(
            "diff",
            SimilarityHeuristic,
            diff_heuristic_runs,
        ),
        RunConfig(
            "largest_first",
            LargestFirstHeuristic,
            largest_first_runs,
        ),
        RunConfig(
            "greedy_ratio",
            GreedyRatioHeuristic,
            greedy_ratio_runs,
        ),
        RunConfig(
            "local_search",
            LocalSearchHeuristic,
            local_search_runs,
        ),
        RunConfig(
            "grasp",
            GRASPHeuristic,
            grasp_runs,
        ),
    ]

    for solver_config in solver_configs:
        process(solver_config, input_folder, output_folder)
