from typing import Callable
from impl.simple_heuristic import SimpleHeuristic
from impl.similarity_heuristic import SimilarityHeuristic
from impl.aisle_first import AisleFirstHeuristic
import numpy as np
import statistics


import time


from models.solver import Solver
import os

from utils.generate_output import generate_output
from utils.read_input import read_input
from utils.wave_order_picking import WaveOrderPicking

import csv


class RunConfig:
    def __init__(
        self,
        name: str,
        solver_class: type[Solver],
        configs: dict,
    ):
        self.name = name
        self.solver_class = solver_class
        self.configs = configs


RUNS = 30


def process(solver_config: RunConfig, input_folder: str, output_folder: str):

    objectives_dir = os.path.join(os.path.dirname(__file__), "..", "objectives_output")
    os.makedirs(objectives_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.normpath(input_folder))
    results = []

    for filename in sorted(os.listdir(input_folder)):

        if not filename.endswith(".txt"):
            continue

        input_file = os.path.join(input_folder, filename)

        wave_order_picking = WaveOrderPicking()
        wave_order_picking.read_input(input_file)

        input = read_input(input_file)

        values = []

        for run in range(RUNS):

            solver = solver_config.solver_class(input, solver_config.configs)

            start = time.perf_counter()

            solution = solver.solve()

            end = time.perf_counter()

            selected_orders = solution[0]
            visited_aisles = solution[1]

            is_feasible = wave_order_picking.is_solution_feasible(
                selected_orders, visited_aisles
            )

            objective_value = wave_order_picking.compute_objective_function(
                selected_orders, visited_aisles
            )

            if is_feasible:
                values.append(objective_value)

            print(
                f"{solver_config.name} - {filename} - {run + 1}/{RUNS} - Time: {end - start:.2f}s - Objective: {objective_value} - Feasible: {is_feasible}",
                flush=True,
            )

            run += 1

        if not len(values) > 0:
            continue

        arr_np = np.array(values)

        mean = np.mean(arr_np)
        median = np.median(arr_np)

        mode = statistics.mode(values)

        minimum = np.min(arr_np)
        maximum = np.max(arr_np)

        variance = np.var(arr_np)
        std_dev = np.std(arr_np)
        coef_variation = std_dev / mean

        results.append(
            (
                dataset_name,
                filename,
                mean,
                median,
                mode,
                minimum,
                maximum,
                variance,
                std_dev,
                coef_variation,
            )
        )

    csv_path = os.path.join(
        objectives_dir,
        f"{solver_config.name}_{dataset_name}_{solver_config.name}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "instance",
                "mean",
                "median",
                "mode",
                "min",
                "max",
                "variance",
                "std_dev",
                "coef_variation",
            ]
        )
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":

    input_folder = "datasets/a"
    output_folder = "output"

    solver_configs = [
        RunConfig(
            "simple",
            SimpleHeuristic,
            {},
        ),
        RunConfig(
            "similar",
            SimilarityHeuristic,
            {
                "reverse": True,
            },
        ),
        RunConfig(
            "diff",
            SimilarityHeuristic,
            {
                "reverse": False,
            },
        ),
    ]

    for solver_config in solver_configs:
        process(solver_config, input_folder, output_folder)
