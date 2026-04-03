import numpy as np

from .base import ProblemInput


def is_solution_feasible(
    problem: ProblemInput,
    selected_orders: list[int],
    visited_aisles: list[int],
) -> bool:
    """Check if a solution satisfies wave-size bounds and stock constraints."""
    total_units_picked = 0
    for order in selected_orders:
        total_units_picked += np.sum(list(problem.orders[order].values()))

    if not (problem.lb <= total_units_picked <= problem.ub):
        return False

    required_items = set()
    for order in selected_orders:
        required_items.update(problem.orders[order].keys())

    for item in required_items:
        total_required = sum(
            problem.orders[order].get(item, 0) for order in selected_orders
        )

        total_available = sum(
            problem.aisles[aisle].get(item, 0) for aisle in visited_aisles
        )

        if total_required > total_available:
            return False

    return True


def compute_objective(
    problem: ProblemInput,
    selected_orders: list[int],
    visited_aisles: list[int],
) -> float:
    """Compute objective: total_items_collected / num_aisles_visited."""
    total_units_picked = 0
    for order in selected_orders:
        total_units_picked += np.sum(list(problem.orders[order].values()))

    num_visited_aisles = len(visited_aisles)
    if num_visited_aisles == 0:
        return 0.0

    return total_units_picked / num_visited_aisles
