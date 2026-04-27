import signal
import time
from algorithms.ga.ga_heuristic import GeneticAlgorithm
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.pan_liu.pan_liu_heuristic import PanLiuHeuristic
from algorithms.dinkelbach_alns.algorithm import DinkelbachALNS

from problems.base import ProblemInput, load_instance
from problems.validation import is_solution_feasible, compute_objective

import pandas as pd


# Timeout handler using SIGALRM (Unix-only)
class _Timeout(BaseException):
    pass


def _timeout_handler(_signum, _frame):
    raise _Timeout()


def run_with_timeout(solver, instance, timeout=90):
    """
    Execute solver with timeout. If timeout occurs, returns best partial solution found by the solver.
    Supports solvers with `last_best` attribute (e.g., GA algorithms).
    """
    start = time.perf_counter()
    timed_out = False
    result = None

    if timeout is not None:
        prev_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)

        try:
            result = solver.solve(instance)
        except _Timeout:
            timed_out = True
            # Try to recover partial best solution from solver
            partial = getattr(solver, "last_best", None)
            if isinstance(partial, dict) and partial.get("selected_orders"):
                result = {
                    "selected_orders": partial["selected_orders"],
                    "visited_aisles": partial["visited_aisles"],
                    "objective": partial.get("objective", 0.0),
                }
                print(
                    f"  [!] Timeout after {timeout:.1f}s. Returning partial best: "
                    f"obj={result['objective']:.2f}, orders={len(result['selected_orders'])}, "
                    f"aisles={len(result['visited_aisles'])}"
                )
            else:
                print(f"  [!] Timeout after {timeout:.1f}s. No partial solution found.")
                result = None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev_handler)
    else:
        result = solver.solve(instance)

    elapsed = time.perf_counter() - start
    return result, timed_out, elapsed


if __name__ == "__main__":
    instance_path = "datasets/a/instance_0007.txt"
    instance = load_instance(instance_path)

    TIMEOUT_SECONDS = 180  # 3 minutes per solver

    solvers = [
        # {
        #     "name": "bigger",
        #     "solver": SimpleHeuristic(
        #         {
        #             "order": "desc",
        #             "greedy": "multi",
        #             "first_order": "bigger",
        #             "exact": True,
        #         }
        #     ),
        # },
        # {
        #     "name": "pan_liu",
        #     "solver": PanLiuHeuristic(
        #         {
        #             "greedy": "multi",
        #         }
        #     ),
        # },
        {
            "name": "aisle_first",
            "solver": AisleFirstHeuristic(
                {"score": "useful", "order": "desc", "prune": "multi"}
            ),
        },
        # {
        #     "name": "ga_full",
        #     "solver": GeneticAlgorithm(
        #         {
        #             "variant": "BaseGA",
        #             "pop_size": 50,
        #             "epoch": 300,
        #             "pc": 0.9,
        #             "pm": 0.05,
        #             "selection": "tournament",
        #             "crossover": "uniform",
        #             "mutation": "flip",
        #             "k_way": 0.2,
        #             "seed_with_heuristics": True,
        #             "start": "seed_aisle",
        #             "local_search": {
        #                 "mode": "final",
        #                 "operators": ["remove", "swap", "add"],
        #                 "strategy": "first_improvement",
        #                 "max_iterations": 400,
        #                 "time_fraction": 0.25,
        #                 "neighbor_cap": 80,
        #             },
        #             "time_budget": TIMEOUT_SECONDS
        #             * 0.95,  # Reserve 95% of timeout for GA evolution
        #         }
        #     ),
        # },
        {
            "name": "dinkelbach_alns",
            "solver": DinkelbachALNS(
                {
                    "time_limit": TIMEOUT_SECONDS
                    * 0.95,  # Reserve 95% of timeout for ALNS search
                    "seed": 42,
                }
            ),
        },
    ]

    for solver_info in solvers:
        print(f"\n{'='*60}")
        print(f"Running solver: {solver_info['name']}")
        print(f"{'='*60}")

        result, timed_out, elapsed = run_with_timeout(
            solver_info["solver"], instance, timeout=TIMEOUT_SECONDS
        )

        if result is not None:
            obj = result.get("objective", 0.0)
            n_orders = len(result.get("selected_orders", []))
            n_aisles = len(result.get("visited_aisles", []))
            status = "[TIMEOUT]" if timed_out else "[OK]"
            print(
                f"{status} {solver_info['name']}: obj={obj:.2f}, "
                f"orders={n_orders}, aisles={n_aisles}, time={elapsed:.2f}s"
            )
        else:
            print(
                f"[FAILED] {solver_info['name']}: No solution found (time={elapsed:.2f}s)"
            )

    df = pd.read_csv("best_solutions/best_objectives.csv")

    df = df[df["instance"] == instance_path.split("/")[-1]]

    print(df)
