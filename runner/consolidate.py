"""Consolidate JSONL results into a summary CSV."""

import json
import os
import statistics

import numpy as np
import pandas as pd


def _build_stats(values):
    """Compute mean, median, mode, min, max, variance, std_dev for a list of numbers."""
    arr = np.array(values)
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "mode": statistics.mode(values),
        "min": np.min(arr),
        "max": np.max(arr),
        "variance": np.var(arr),
        "std_dev": np.std(arr),
    }


def _to_set(value):
    """Return a set from a list-like JSON field, else an empty set."""
    if isinstance(value, list):
        return set(value)
    return set()


def _jaccard_similarity(a, b):
    """Compute Jaccard index for two sets; returns 1.0 when both are empty."""
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def consolidate_results(
    jsonl_path: str, output_dir: str, save_output: bool = False
) -> str:
    """Read JSONL and write summary.csv. Returns path to CSV."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Results file not found: {jsonl_path}")

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in results file.")
        return ""

    df = pd.DataFrame(records)

    os.makedirs(output_dir, exist_ok=True)

    # Derive unique suffix from JSONL filename (e.g. runs_a_20260330_120000.jsonl → a_20260330_120000)
    jsonl_stem = os.path.splitext(os.path.basename(jsonl_path))[0]
    suffix = (
        jsonl_stem.removeprefix("runs_")
        if jsonl_stem.startswith("runs_")
        else jsonl_stem
    )

    # Build aggregated summary (only feasible runs)
    csv_path = os.path.join(output_dir, f"summary_{suffix}.csv")

    if "algorithm" not in df.columns or "objective" not in df.columns:
        print("Missing required columns (algorithm, objective). No summary generated.")
        return ""

    feasible = df[df["feasible"] == True]

    if feasible.empty:
        print("No feasible runs found. No summary generated.")
        return ""

    # Determine dataset column
    dataset_col = "dataset" if "dataset" in feasible.columns else None

    summary_rows = []
    group_cols = ["algorithm", "instance"]
    if dataset_col:
        group_cols.insert(0, dataset_col)

    for group_key, group_df in feasible.groupby(group_cols):
        if dataset_col:
            dataset, algorithm, instance = group_key
        else:
            algorithm, instance = group_key
            # Derive dataset from instance_path if available
            if "instance_path" in group_df.columns:
                first_path = group_df["instance_path"].iloc[0]
                dataset = os.path.basename(os.path.dirname(first_path))
            else:
                dataset = ""

        total_runs = len(
            df[(df["algorithm"] == algorithm) & (df["instance"] == instance)]
        )
        feasible_runs = len(group_df)
        timed_out_runs = (
            int(
                df[
                    (df["algorithm"] == algorithm)
                    & (df["instance"] == instance)
                    & (df.get("timed_out", pd.Series(False)) == True)
                ].shape[0]
            )
            if "timed_out" in df.columns
            else 0
        )

        obj_stats = _build_stats(group_df["objective"].tolist())

        items_values = (
            group_df["total_items"].tolist()
            if "total_items" in group_df.columns
            else []
        )
        items_stats = (
            _build_stats(items_values)
            if items_values
            else {
                k: 0
                for k in ["mean", "median", "mode", "min", "max", "variance", "std_dev"]
            }
        )

        aisles_values = (
            group_df["num_aisles"].tolist() if "num_aisles" in group_df.columns else []
        )
        aisles_stats = (
            _build_stats(aisles_values)
            if aisles_values
            else {
                k: 0
                for k in ["mean", "median", "mode", "min", "max", "variance", "std_dev"]
            }
        )

        time_col = (
            "time_s"
            if "time_s" in group_df.columns
            else "exec_time" if "exec_time" in group_df.columns else None
        )
        time_values = group_df[time_col].tolist() if time_col else []
        time_stats = (
            _build_stats(time_values)
            if time_values
            else {
                k: 0
                for k in ["mean", "median", "mode", "min", "max", "variance", "std_dev"]
            }
        )

        row = {
            "dataset": dataset,
            "algorithm": algorithm,
            "instance": instance,
            "total_runs": total_runs,
            "feasible_runs": feasible_runs,
            "timed_out_runs": timed_out_runs,
        }
        for prefix, stats in [
            ("objective", obj_stats),
            ("items", items_stats),
            ("aisles", aisles_stats),
            ("exec_time", time_stats),
        ]:
            for stat_name, stat_value in stats.items():
                row[f"{prefix}_{stat_name}"] = stat_value

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csv_path, index=False)

    print(
        f"Consolidated {len(records)} records → {len(summary_rows)} summary rows → {csv_path}"
    )

    # Print summary to console
    print("\n--- Summary (feasible runs) ---")
    display_cols = [
        "algorithm",
        "instance",
        "feasible_runs",
        "objective_mean",
        "objective_max",
        "items_mean",
        "aisles_mean",
        "exec_time_mean",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    if not save_output:
        return csv_path

    # Build per-instance method similarity and selection distribution tables.
    required_cols = {
        "algorithm",
        "instance",
        "selected_orders",
        "visited_aisles",
        "selected_items",
    }
    if required_cols.issubset(df.columns):
        best_idx = df.groupby(["algorithm", "instance"])["objective"].idxmax()
        best_df = df.loc[best_idx].copy()

        similarity_rows = []
        distribution_rows = []

        for instance, instance_df in best_df.groupby("instance"):
            rows = list(instance_df.to_dict(orient="records"))

            for row in rows:
                distribution_rows.append(
                    {
                        "instance": instance,
                        "algorithm": row["algorithm"],
                        "num_orders": len(row.get("selected_orders", [])),
                        "num_aisles": len(row.get("visited_aisles", [])),
                        "num_items": len(row.get("selected_items", [])),
                        "selected_orders": "|".join(
                            map(str, row.get("selected_orders", []))
                        ),
                        "visited_aisles": "|".join(
                            map(str, row.get("visited_aisles", []))
                        ),
                        "selected_items": "|".join(
                            map(str, row.get("selected_items", []))
                        ),
                        "objective": row.get("objective", 0.0),
                        "feasible": row.get("feasible", False),
                    }
                )

            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    left = rows[i]
                    right = rows[j]
                    left_orders = _to_set(left.get("selected_orders"))
                    right_orders = _to_set(right.get("selected_orders"))
                    left_aisles = _to_set(left.get("visited_aisles"))
                    right_aisles = _to_set(right.get("visited_aisles"))
                    left_items = _to_set(left.get("selected_items"))
                    right_items = _to_set(right.get("selected_items"))

                    similarity_rows.append(
                        {
                            "instance": instance,
                            "algorithm_left": left["algorithm"],
                            "algorithm_right": right["algorithm"],
                            "orders_jaccard": _jaccard_similarity(
                                left_orders, right_orders
                            ),
                            "aisles_jaccard": _jaccard_similarity(
                                left_aisles, right_aisles
                            ),
                            "items_jaccard": _jaccard_similarity(
                                left_items, right_items
                            ),
                            "orders_overlap": len(left_orders & right_orders),
                            "aisles_overlap": len(left_aisles & right_aisles),
                            "items_overlap": len(left_items & right_items),
                        }
                    )

        if similarity_rows:
            similarity_path = os.path.join(output_dir, f"similarity_{suffix}.csv")
            pd.DataFrame(similarity_rows).to_csv(similarity_path, index=False)
            print(f"Similarity CSV: {similarity_path}")

        if distribution_rows:
            distribution_path = os.path.join(output_dir, f"distribution_{suffix}.csv")
            pd.DataFrame(distribution_rows).to_csv(distribution_path, index=False)
            print(f"Distribution CSV: {distribution_path}")

    return csv_path
