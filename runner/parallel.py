"""Parallel experiment execution using ProcessPoolExecutor."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from runner.experiment import run_task


def build_task_list(
    config: dict,
    instance_paths: list[str],
    result_path: str,
) -> list[dict]:
    """Build a flat list of tasks from config and instance paths."""
    tasks = []
    repetitions = config.get("repetitions", 1)
    seed_base = config.get("seed_base", 42)
    time_limit = config.get("time_limit", None)

    for algo_cfg in config["algorithms"]:
        algo_name = algo_cfg["name"]
        algo_key = algo_cfg.get("algo", algo_name)
        algo_params = algo_cfg.get("params", {})

        is_deterministic = algo_cfg.get("deterministic", False)

        for inst_path in instance_paths:

            if is_deterministic:
                seed = seed_base
                tasks.append(
                    {
                        "algo_name": algo_name,
                        "algo_key": algo_key,
                        "algo_params": algo_params,
                        "instance_path": inst_path,
                        "run_id": 0,
                        "seed": seed,
                        "result_path": result_path,
                        "time_limit": time_limit,
                    }
                )
                continue

            for run_id in range(repetitions):
                seed = seed_base + run_id
                tasks.append(
                    {
                        "algo_name": algo_name,
                        "algo_key": algo_key,
                        "algo_params": algo_params,
                        "instance_path": inst_path,
                        "run_id": run_id,
                        "seed": seed,
                        "result_path": result_path,
                        "time_limit": time_limit,
                    }
                )

    return tasks


def run_all(
    config: dict,
    instance_paths: list[str],
    result_path: str,
) -> list[dict]:
    """Run all experiments in parallel and return results."""
    tasks = build_task_list(config, instance_paths, result_path)

    if not tasks:
        print("No tasks to run.")
        return []

    max_workers = config.get("max_workers")
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 1) - 1)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_task,
                task["algo_name"],
                task["algo_key"],
                task["algo_params"],
                task["instance_path"],
                task["run_id"],
                task["seed"],
                task["result_path"],
                task["time_limit"],
            ): task
            for task in tasks
        }

        with tqdm(total=len(futures), desc="Running experiments") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    status = "OK" if result["feasible"] else "INFEASIBLE"
                    if result["timed_out"]:
                        status = "TIMEOUT"

                    pbar.set_postfix_str(
                        f"{result['algorithm']}/{result['instance']} "
                        f"obj={result['objective']:.2f} [{status}]"
                    )
                except Exception as exc:
                    algo = task["algo_name"]
                    inst = os.path.basename(task["instance_path"])
                    tqdm.write(f"FAILED: {algo}/{inst} — {exc}")

                pbar.update(1)

    return results
