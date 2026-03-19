from typing import Callable
from impl.simple_heuristic import SimpleHeuristic
from impl.similarity_heuristic import SimilarityHeuristic

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


def simple_heuristic_runs(input: ProblemInput) -> list[dict]:

    base = list(range(0, input.nOrders))

    configs = []

    seeds = [random.sample(base, len(base)) for _ in range(runs)]

    for seed in seeds:
        configs.append({"seed": seed})

    return configs


def similar_heuristic_runs(input: ProblemInput) -> list[dict]:

    base = list(range(0, input.nOrders))

    configs = []

    seeds = [random.sample(base, len(base)) for _ in range(runs)]

    for seed in seeds:
        configs.append({"seed": seed, "reverse": True})

    return configs


def diff_heuristic_runs(input: ProblemInput) -> list[dict]:

    base = list(range(0, input.nOrders))

    configs = []

    seeds = [random.sample(base, len(base)) for _ in range(runs)]

    for seed in seeds:
        configs.append({"seed": seed, "reverse": False})

    return configs


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
    ]

    for solver_config in solver_configs:
        process(solver_config, input_folder, output_folder)
