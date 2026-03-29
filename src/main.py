from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from impl.simple_heuristic import SimpleHeuristic
from impl.similarity_heuristic import SimilarityHeuristic
from impl.aisle_first import AisleFirstHeuristic
from impl.local_search import LocalSearchHeuristic
from impl.simulated_annealing import SimulatedAnnealingHeuristic
from impl.max_density_order_insertion import MaxDensityOrderInsertionHeuristic
from impl.greedy_order_similarity_aggregation import (
    GreedyOrderSimilarityAggregationHeuristic,
)
from impl.greedy_aisle_density import GreedyAisleDensityHeuristic
from impl.aisle_cluster_expansion import AisleClusterExpansion
from impl.ping_pong_alternating import PingPongAlternatingHeuristic
from impl.grasp_heuristic import GRASPHeuristic
from impl.genetic_algorithm_heuristic import GeneticAlgorithmHeuristic
from impl.ant_colony_optimization import AntColonyOptimizationHeuristic
from impl.tabu_search_heuristic import TabuSearchHeuristic
from impl.simulated_annealing_heuristic import SimulatedAnnealingHeuristic
from impl.dinkelbach_milp import DinkelbachMILPSolver
import numpy as np
import statistics


import time


from models.solver import Solver
import os
import signal

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


RUNS = 2
SOLVER_TIME_LIMIT_SECONDS = 5.0


class SolverTimeoutError(Exception):
    pass


def _on_solver_timeout(_signum, _frame):
    raise SolverTimeoutError()


def _solve_with_timeout(solver):
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _on_solver_timeout)
    signal.setitimer(signal.ITIMER_REAL, SOLVER_TIME_LIMIT_SECONDS)

    try:
        return solver.solve()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _build_stats(values: list[float]):
    arr_np = np.array(values)

    return {
        "mean": np.mean(arr_np),
        "median": np.median(arr_np),
        "mode": statistics.mode(values),
        "min": np.min(arr_np),
        "max": np.max(arr_np),
        "variance": np.var(arr_np),
        "std_dev": np.std(arr_np),
    }


def _compute_instance_stats(
    dataset_name: str,
    filename: str,
    solver_class: type[Solver],
    solver_configs: dict,
    problem_input,
):
    wave_order_picking = WaveOrderPicking()
    wave_order_picking.load_problem_input(problem_input)

    objective_values = []
    item_count_values = []
    aisle_count_values = []
    exec_time_values = []
    total_elapsed = 0.0
    timed_out_runs = 0

    for _ in range(RUNS):
        solver = solver_class(problem_input, solver_configs)

        start = time.perf_counter()
        try:
            selected_orders, visited_aisles = _solve_with_timeout(solver)
        except SolverTimeoutError:
            timed_out_runs += 1
            elapsed = min(
                time.perf_counter() - start,
                SOLVER_TIME_LIMIT_SECONDS,
            )
            total_elapsed += elapsed
            continue

        elapsed = time.perf_counter() - start
        total_elapsed += elapsed

        is_feasible = wave_order_picking.is_solution_feasible(
            selected_orders, visited_aisles
        )
        objective_value = wave_order_picking.compute_objective_function(
            selected_orders, visited_aisles
        )

        if is_feasible:
            total_items = sum(
                sum(problem_input.orders[order].values()) for order in selected_orders
            )

            objective_values.append(objective_value)
            item_count_values.append(total_items)
            aisle_count_values.append(len(visited_aisles))
            exec_time_values.append(elapsed)

    if not objective_values:
        return {
            "filename": filename,
            "row": None,
            "summary": (
                f"{filename} - feasible runs: 0/{RUNS} - timeouts: {timed_out_runs}/{RUNS} - "
                f"total solve time: {total_elapsed:.2f}s"
            ),
        }

    objective_stats = _build_stats(objective_values)
    item_count_stats = _build_stats(item_count_values)
    aisle_count_stats = _build_stats(aisle_count_values)
    exec_time_stats = _build_stats(exec_time_values)

    row = (
        dataset_name,
        filename,
        objective_stats["mean"],
        objective_stats["median"],
        objective_stats["mode"],
        objective_stats["min"],
        objective_stats["max"],
        objective_stats["variance"],
        objective_stats["std_dev"],
        item_count_stats["mean"],
        item_count_stats["median"],
        item_count_stats["mode"],
        item_count_stats["min"],
        item_count_stats["max"],
        item_count_stats["variance"],
        item_count_stats["std_dev"],
        aisle_count_stats["mean"],
        aisle_count_stats["median"],
        aisle_count_stats["mode"],
        aisle_count_stats["min"],
        aisle_count_stats["max"],
        aisle_count_stats["variance"],
        aisle_count_stats["std_dev"],
        exec_time_stats["mean"],
        exec_time_stats["median"],
        exec_time_stats["mode"],
        exec_time_stats["min"],
        exec_time_stats["max"],
        exec_time_stats["variance"],
        exec_time_stats["std_dev"],
    )

    summary = (
        f"{filename} - feasible runs: {len(objective_values)}/{RUNS} - "
        f"timeouts: {timed_out_runs}/{RUNS} - "
        f"objective mean: {objective_stats['mean']:.4f} - "
        f"items mean: {item_count_stats['mean']:.2f} - "
        f"aisles mean: {aisle_count_stats['mean']:.2f} - "
        f"exec mean: {exec_time_stats['mean']:.4f}s - total solve time: {total_elapsed:.2f}s"
    )

    return {"filename": filename, "row": row, "summary": summary}


def preload_instances(input_folder: str):
    instances = []

    for filename in sorted(os.listdir(input_folder)):
        if not filename.endswith(".txt"):
            continue

        input_file = os.path.join(input_folder, filename)
        problem_input = read_input(input_file)

        instances.append((filename, problem_input))

    return instances


def process(
    solver_config: RunConfig,
    input_folder: str,
    output_folder: str,
    instances,
    max_workers: int | None = None,
):

    objectives_dir = os.path.join(os.path.dirname(__file__), "..", "objectives_output")
    os.makedirs(objectives_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.normpath(input_folder))
    results = []
    instance_reports = []

    if not instances:
        print(f"No instances found in {input_folder}")
        return

    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _compute_instance_stats,
                dataset_name,
                filename,
                solver_config.solver_class,
                solver_config.configs,
                problem_input,
            )
            for filename, problem_input in instances
        ]

        for future in as_completed(futures):
            report = future.result()
            instance_reports.append(report)

    for report in sorted(instance_reports, key=lambda item: item["filename"]):
        print(f"{solver_config.name} - {report['summary']}", flush=True)
        if report["row"] is not None:
            results.append(report["row"])

    csv_path = os.path.join(
        objectives_dir,
        f"{solver_config.name}_{dataset_name}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "instance",
                "objective_mean",
                "objective_median",
                "objective_mode",
                "objective_min",
                "objective_max",
                "objective_variance",
                "objective_std_dev",
                "items_mean",
                "items_median",
                "items_mode",
                "items_min",
                "items_max",
                "items_variance",
                "items_std_dev",
                "aisles_mean",
                "aisles_median",
                "aisles_mode",
                "aisles_min",
                "aisles_max",
                "aisles_variance",
                "aisles_std_dev",
                "exec_time_mean",
                "exec_time_median",
                "exec_time_mode",
                "exec_time_min",
                "exec_time_max",
                "exec_time_variance",
                "exec_time_std_dev",
            ]
        )
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")


def run_solver_job(
    solver_config: RunConfig,
    input_folder: str,
    output_folder: str,
    instances,
    max_workers: int,
):
    start = time.perf_counter()
    process(
        solver_config=solver_config,
        input_folder=input_folder,
        output_folder=output_folder,
        instances=instances,
        max_workers=max_workers,
    )
    elapsed = time.perf_counter() - start

    return {
        "name": solver_config.name,
        "elapsed": elapsed,
    }


if __name__ == "__main__":

    input_folder = "datasets/a"
    output_folder = "output"
    total_cpu_workers = max(1, (os.cpu_count() or 1) - 1)

    instances = preload_instances(input_folder)

    solver_configs = [
        # RunConfig(
        #     "simple",
        #     SimpleHeuristic,
        #     {
        #         "greedy": "simple",
        #     },
        # ),
        # RunConfig(
        #     "simple_multi",
        #     SimpleHeuristic,
        #     {
        #         "greedy": "multi",
        #     },
        # ),
        # RunConfig(
        #     "similar",
        #     SimilarityHeuristic,
        #     {
        #         "reverse": True,
        #         "greedy": "simple",
        #     },
        # ),
        # RunConfig(
        #     "similar_multi",
        #     SimilarityHeuristic,
        #     {
        #         "reverse": True,
        #         "greedy": "multi",
        #     },
        # ),
        # RunConfig(
        #     "diff",
        #     SimilarityHeuristic,
        #     {
        #         "reverse": False,
        #         "greedy": "simple",
        #     },
        # ),
        # RunConfig(
        #     "diff_multi",
        #     SimilarityHeuristic,
        #     {
        #         "reverse": False,
        #         "greedy": "multi",
        #     },
        # ),
        # RunConfig(
        #     "similarity_aggregation_multi",
        #     GreedyOrderSimilarityAggregationHeuristic,
        #     {},
        # ),
        # RunConfig(
        #     "ga_balanced",
        #     GeneticAlgorithmHeuristic,
        #     {
        #         "population_size": 80,
        #         "generations": 120,
        #         "crossover_rate": 0.85,
        #         "mutation_rate": 0.02,
        #         "tournament_size": 3,
        #         "elite_count": 4,
        #         "penalty_scale": 3.0,
        #         "penalty_lb": 1.0,
        #         "penalty_ub": 1.0,
        #         "penalty_stock": 2.0,
        #         "repair_upper_bound": True,
        #         "feasible_init_ratio": 0.6,
        #     },
        # ),
        # RunConfig(
        #     "aco_pairwise",
        #     AntColonyOptimizationHeuristic,
        #     {
        #         "n_ants": 30,
        #         "n_iterations": 60,
        #         "alpha": 1.1,
        #         "beta": 2.2,
        #         "evaporation": 0.15,
        #         "q_deposit": 1.0,
        #         "initial_pheromone": 1.0,
        #         "candidate_list_size": 25,
        #         "stop_after_lb_prob": 0.2,
        #         "elitist_weight": 2.0,
        #     },
        # ),
        # RunConfig(
        #     "tabu_search",
        #     TabuSearchHeuristic,
        #     {
        #         "max_iterations": 180,
        #         "no_improve_limit": 60,
        #         "neighborhood_samples": 40,
        #         "tabu_tenure_orders": 10,
        #         "tabu_tenure_aisles": 8,
        #     },
        # ),
        # RunConfig(
        #     "sa_basic",
        #     SimulatedAnnealingHeuristic,
        #     {
        #         "max_iterations": 5000,
        #         "max_time_seconds": 1.5,
        #         "initial_temp": 2.5,
        #         "cooling_rate": 0.9975,
        #         "min_temp": 1e-4,
        #         "neighbor_tries": 40,
        #         "initial_attempts": 24,
        #     },
        # ),
        # RunConfig(
        #     "local_search",
        #     LocalSearchHeuristic,
        #     {
        #         "iterations": 10,
        #         "max_no_improve": 5,
        #     },
        # ),
        # RunConfig(
        #     "simulated_annealing",
        #     SimulatedAnnealingHeuristic,
        #     {
        #         "T_init": 5.0,
        #         "T_min": 0.001,
        #         "alpha": 0.998,
        #         "max_iter": 5000,
        #     },
        # ),
        RunConfig(
            "dinkelbach_milp",
            DinkelbachMILPSolver,
            {
                "max_iterations": 20,
                "epsilon": 1e-6,
                "time_limit": 2.0,
                "total_time_limit": 4.5,
            },
        ),
        # RunConfig(
        #     "max_density_insert",
        #     MaxDensityOrderInsertionHeuristic,
        #     {},
        # ),
        # RunConfig(
        #     "greedy_aisle_density",
        #     GreedyAisleDensityHeuristic,
        #     {
        #         "initial_k": 1,
        #     },
        # ),
        # RunConfig(
        #     "aisle_cluster_expansion",
        #     AisleClusterExpansion,
        #     {
        #         "attempts": 20,
        #         "max_added_aisles": 20,
        #     },
        # ),
        # RunConfig(
        #     "ping_pong_alternating",
        #     PingPongAlternatingHeuristic,
        #     {
        #         "phase1_mode": "both",
        #         "max_iterations": 25,
        #         "min_improvement_delta": 1e-5,
        #     },
        # ),
        # RunConfig(
        #     "grasp",
        #     GRASPHeuristic,
        #     {
        #         "max_iterations": 50,
        #         "alpha": 0.3,
        #         "local_search_no_improve": 8,
        #         "construction_stagnation": 4,
        #         "max_order_swap_checks": 20000,
        #         "max_aisle_swap_checks": 5000,
        #     },
        # ),
    ]

    if not solver_configs:
        raise ValueError("No solver configured in solver_configs")

    solver_parallelism = min(len(solver_configs), total_cpu_workers)
    workers_per_solver = max(1, total_cpu_workers // solver_parallelism)

    print(
        f"Running {len(solver_configs)} solver(s) in parallel (max {solver_parallelism} at a time) "
        f"with up to {workers_per_solver} process worker(s) per solver"
    )

    with ThreadPoolExecutor(max_workers=solver_parallelism) as executor:
        future_to_solver = {}

        for solver_config in solver_configs:
            print(
                f"Scheduling solver: {solver_config.name}, {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            future = executor.submit(
                run_solver_job,
                solver_config,
                input_folder,
                output_folder,
                instances,
                workers_per_solver,
            )
            future_to_solver[future] = solver_config.name

        for future in as_completed(future_to_solver):
            solver_name = future_to_solver[future]
            try:
                result = future.result()
                print(
                    f"Finished solver: {result['name']} in {result['elapsed']:.2f}s, "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            except Exception as exc:
                print(f"Solver failed: {solver_name} with error: {exc}")
