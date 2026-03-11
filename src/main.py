from impl.greedy import Greedy
from impl.aisle_centric import AisleCentric
from impl.order_clustering import OrderClustering
from models.solver import Solver
import os
from sys import argv

from utils.generate_output import generate_output
from utils.read_input import readInput
from utils.wave_order_picking import WaveOrderPicking

import csv


def solve(output_file, solver):
    solution = solver.solve()

    generate_output(output_file, len(solution[0]), len(
        solution[1]), solution[0], solution[1])


def process(solver_class):
    input_folder = 'datasets/a'
    output_folder = 'output'

    dataset_name = os.path.basename(os.path.normpath(input_folder))
    results = []

    for filename in sorted(os.listdir(input_folder)):

        if not filename.endswith('.txt'):
            continue

        print(f"Running {solver_class.__name__} on {filename}", flush=True)

        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)

        nOrders, nItems, nAisles, orders, aisles, lb, ub = readInput(input_file)
        solver = solver_class(nOrders, nAisles, orders, aisles, lb, ub)

        solve(output_file, solver)

        wave_order_picking = WaveOrderPicking()
        wave_order_picking.read_input(input_file)
        selected_orders, visited_aisles = wave_order_picking.read_output(
            output_file)

        is_feasible = wave_order_picking.is_solution_feasible(
            selected_orders, visited_aisles)

        objective_value = wave_order_picking.compute_objective_function(
            selected_orders, visited_aisles)

        print("Is solution feasible:", is_feasible)

        if is_feasible:
            print("Objective function value:", objective_value)
            results.append((dataset_name, filename, objective_value))
        else:
            results.append((dataset_name, filename, -1))

    objectives_dir = os.path.join(os.path.dirname(__file__), '..', 'objectives')
    os.makedirs(objectives_dir, exist_ok=True)

    csv_path = os.path.join(objectives_dir, f'{solver_class.__name__}_{dataset_name}_objectives.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'instance', 'best_objective'])
        writer.writerows(results)

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":

    solvers = [
        Greedy,
        AisleCentric,
        OrderClustering,
    ]

    for solver_class in solvers:
        process(solver_class)

