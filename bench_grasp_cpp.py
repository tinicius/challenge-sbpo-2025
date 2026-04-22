"""Benchmark: Python GraspHeuristic vs GraspCppHeuristic on a mix of instances."""

import time

from algorithms.grasp.grasp_cpp_heuristic import GraspCppHeuristic
from algorithms.grasp.grasp_heuristic import GraspHeuristic
from problems.base import load_instance
from problems.validation import is_solution_feasible, compute_objective


CONFIGS = [
    {"alpha": 0.3, "construction_score": "size",
     "max_iterations": 20, "greedy": "multi", "local_search": "swap"},
    {"alpha": 0.3, "construction_score": "synergy",
     "max_iterations": 10, "greedy": "multi", "local_search": "full"},
    {"alpha": 0.2, "construction_score": "aisle_cost_fast",
     "max_iterations": 10, "greedy": "multi", "local_search": "full"},
]

# Python is orders of magnitude slower, so we stick to small/mid instances to
# keep the benchmark under a few minutes. The C++ side can handle much larger
# inputs within the same budget.
INSTANCES = [
    "datasets/a/instance_0002.txt",  # tiny
    "datasets/a/instance_0004.txt",  # small
    "datasets/a/instance_0001.txt",  # small
    "datasets/a/instance_0009.txt",  # small-mid
    "datasets/a/instance_0012.txt",  # mid
]

SEED = 42


def run(algo, inst):
    t = time.perf_counter()
    r = algo.solve(inst)
    el = time.perf_counter() - t
    feas = is_solution_feasible(inst, r["selected_orders"], r["visited_aisles"])
    obj = compute_objective(inst, r["selected_orders"], r["visited_aisles"]) if feas else 0.0
    return el, obj, feas


def header():
    print(f"{'instance':<32} {'config':<22} "
          f"{'py_s':>8} {'cpp_s':>8} {'speedup':>8} "
          f"{'py_obj':>9} {'cpp_obj':>9}")


def row(inst, cfg_label, py_s, cpp_s, py_obj, cpp_obj):
    speedup = py_s / cpp_s if cpp_s > 0 else float("inf")
    print(f"{inst:<32} {cfg_label:<22} "
          f"{py_s:>8.2f} {cpp_s:>8.2f} {speedup:>8.1f}x "
          f"{py_obj:>9.3f} {cpp_obj:>9.3f}")


def main():
    header()
    tot_py = tot_cpp = 0.0
    for inst_path in INSTANCES:
        inst = load_instance(inst_path)
        for cfg in CONFIGS:
            params = dict(cfg, seed=SEED)
            label = f"{cfg['construction_score']}/{cfg['local_search']}"
            py_s, py_obj, py_feas = run(GraspHeuristic(params), inst)
            cpp_s, cpp_obj, cpp_feas = run(GraspCppHeuristic(params), inst)
            assert py_feas and cpp_feas, (inst_path, label, py_feas, cpp_feas)
            tot_py += py_s
            tot_cpp += cpp_s
            row(inst_path.split("/")[-1], label, py_s, cpp_s, py_obj, cpp_obj)
    print("-" * 100)
    print(f"totals: python={tot_py:.2f}s cpp={tot_cpp:.2f}s speedup={tot_py/tot_cpp:.1f}x")


if __name__ == "__main__":
    main()
