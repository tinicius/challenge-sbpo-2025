"""Orchestrates a single experiment run (one algorithm × one instance × one seed)."""

import json
import os
import time
from pathlib import Path

from problems.base import ProblemInput, load_instance
from problems.validation import is_solution_feasible, compute_objective


def run_task(
    algo_name: str,
    algo_key: str,
    algo_params: dict,
    instance_path: str,
    run_id: int,
    seed: int,
    result_path: str,
    time_limit: float | None = None,
) -> dict:
    """Execute one algorithm run on one instance. Writes result to JSONL."""
    from algorithms.registry import REGISTRY

    instance = load_instance(instance_path)
    algo_cls = REGISTRY[algo_key]
    algo = algo_cls(algo_params)

    start = time.perf_counter()
    timed_out = False

    if time_limit is not None:
        import signal

        class _Timeout(Exception):
            pass

        def _handler(_signum, _frame):
            raise _Timeout()

        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, time_limit)

        try:
            result = algo.solve(instance)
        except _Timeout:
            timed_out = True
            result = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)
    else:
        result = algo.solve(instance)

    elapsed = time.perf_counter() - start

    selected_orders = result.get("selected_orders", [])
    visited_aisles = result.get("visited_aisles", [])
    objective = result.get("objective", 0.0)

    feasible = False
    if selected_orders and visited_aisles:
        feasible = is_solution_feasible(instance, selected_orders, visited_aisles)
        if feasible:
            objective = compute_objective(instance, selected_orders, visited_aisles)

    total_items = 0
    selected_items = set()
    if selected_orders:
        total_items = sum(sum(instance.orders[o].values()) for o in selected_orders)
        for o in selected_orders:
            selected_items.update(instance.orders[o].keys())

    record = {
        "algorithm": algo_name,
        "instance": os.path.basename(instance_path),
        "instance_path": instance_path,
        "run_id": run_id,
        "seed": seed,
        "objective": float(objective),
        "time_s": round(elapsed, 6),
        "feasible": feasible,
        "timed_out": timed_out,
        "total_items": total_items,
        "num_aisles": len(visited_aisles),
        "num_orders": len(selected_orders),
        "selected_orders": selected_orders,
        "visited_aisles": visited_aisles,
        "selected_items": sorted(selected_items),
        "params": algo_params,
    }

    append_jsonl(result_path, record)
    return record


def append_jsonl(path: str, record: dict) -> None:
    """Append a single JSON record to a JSONL file (crash-safe)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())
