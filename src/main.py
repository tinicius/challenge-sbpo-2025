from impl.greedy import Greedy
from models.solver import Solver
import os
from sys import argv

from utils.generate_output import generate_output
from utils.read_input import readInput
from utils.wave_order_picking import WaveOrderPicking


def solve(input_file, output_file):
    nOrders, nItems, nAisles, orders, aisles, lb, ub = readInput(input_file)

    solver: Solver = Greedy(nOrders, nAisles, orders, aisles, lb, ub)

    solution = solver.solve()

    generate_output(output_file, len(solution[0]), len(
        solution[1]), solution[0], solution[1])


if __name__ == "__main__":
    input_folder = argv[1] if len(argv) > 1 else 'datasets'
    output_folder = argv[2] if len(argv) > 2 else 'outputs'

    for filename in os.listdir(input_folder):

        if not filename.endswith('.txt'):
            continue

        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        solve(input_file, output_file)

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
