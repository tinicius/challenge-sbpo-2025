#!/usr/bin/env python3
"""Entry point: loads config.yaml and runs all experiments."""

import os
import sys
import time
from glob import glob
from pathlib import Path

import yaml


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    instance_dir = config.get("instance_dir", "datasets/a")
    result_dir = config.get("result_dir", "results")

    # Discover instance files
    instance_paths = sorted(glob(os.path.join(instance_dir, "*.txt")))
    if not instance_paths:
        print(f"No .txt instance files found in {instance_dir}")
        sys.exit(1)

    print(f"Found {len(instance_paths)} instances in {instance_dir}")
    print(f"Algorithms: {[a['name'] for a in config['algorithms']]}")
    print(f"Repetitions: {config.get('repetitions', 1)}")
    print(f"Time limit: {config.get('time_limit', 'none')}s")

    total_runs = (
        len(config["algorithms"])
        * len(instance_paths)
        * config.get("repetitions", 1)
    )
    print(f"Total runs: {total_runs}")
    print()

    # JSONL output path — group results in a named subfolder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(instance_dir))
    run_id = f"{dataset_name}_{timestamp}"
    run_dir = os.path.join(result_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    jsonl_path = os.path.join(run_dir, f"runs_{run_id}.jsonl")

    # Run experiments
    from runner.parallel import run_all

    start = time.perf_counter()
    results = run_all(config, instance_paths, jsonl_path)
    elapsed = time.perf_counter() - start

    print(f"\nCompleted {len(results)} runs in {elapsed:.1f}s")
    print(f"JSONL log: {jsonl_path}")

    # Consolidate to CSV
    from runner.consolidate import consolidate_results

    csv_path = consolidate_results(jsonl_path, run_dir)
    if csv_path:
        print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
