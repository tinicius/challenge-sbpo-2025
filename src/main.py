from typing import Callable
from impl.simple_heuristic import SimpleHeuristic
from datetime import datetime, timezone

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

        print(
            f"Running {solver_config.solver_class.__name__} on {filename}", flush=True
        )

        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)

        wave_order_picking = WaveOrderPicking()
        wave_order_picking.read_input(input_file)

        input = read_input(input_file)

        objectives_sum = 0
        number_feasibles_solutions = 0

        for run_config in solver_config.get_runs(input):

            solver = solver_config.solver_class(input, run_config)

            solve(output_file, solver)

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

        results.append(
            (dataset_name, filename, objectives_sum / number_feasibles_solutions)
        )

    csv_path = os.path.join(
        objectives_dir,
        f"{solver_config.solver_class.__name__}_{dataset_name}_{solver_config.name}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "instance", "best_objective"])
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")


def simple_heuristic_runs(input: ProblemInput) -> list[dict]:
    # Each run uses a single randomly chosen order as the seed anchor.
    # The solver then ranks ALL remaining orders by Jaccard similarity to
    # the seed order's item profile and greedily fills the wave from there.
    # Using distinct single-order seeds across runs produces meaningfully
    # different item-profile anchors and therefore genuinely different waves.

    runs = 5

    seed_orders = random.sample(range(input.nOrders), min(runs, input.nOrders))

    return [{"seed": [order_idx]} for order_idx in seed_orders]


if __name__ == "__main__":

    input_folder = "datasets/b"
    output_folder = "output"

    solver_configs = [
        RunConfig(
            "5",
            SimpleHeuristic,
            simple_heuristic_runs,
        ),
    ]

    for solver_config in solver_configs:
        process(solver_config, input_folder, output_folder)
