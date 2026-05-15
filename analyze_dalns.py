"""Compare dinkelbach_alns runs against best-known objectives.

Reads the per-instance JSONL log produced by run_experiments.py and the
best_solutions/best_objectives.csv reference. Prints a per-instance gap table
and a summary by dataset, then highlights instances with > 5% gap.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


BEST_CSV = "best_solutions/best_objectives.csv"


def load_best() -> dict[tuple[str, str], float]:
    best: dict[tuple[str, str], float] = {}
    with open(BEST_CSV) as f:
        next(f)
        for line in f:
            ds, inst, obj = line.strip().split(",")
            best[(ds, inst)] = float(obj)
    return best


def load_runs(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def instance_size(instance_path: str) -> tuple[int, int, int]:
    with open(instance_path) as f:
        n_o, n_i, n_a = (int(x) for x in f.readline().split())
    return n_o, n_i, n_a


def main(jsonl_paths: list[str]) -> None:
    best = load_best()
    rows: list[dict] = []
    for path in jsonl_paths:
        runs = load_runs(path)
        for r in runs:
            ds = os.path.basename(os.path.dirname(r["instance_path"]))
            inst = r["instance"]
            ref = best.get((ds, inst))
            obj = r["objective"]
            gap_pct = (
                100.0 * (ref - obj) / ref if ref and ref > 0 else float("nan")
            )
            try:
                n_o, n_i, n_a = instance_size(r["instance_path"])
            except FileNotFoundError:
                n_o = n_i = n_a = -1
            rows.append({
                "dataset": ds,
                "instance": inst,
                "ref": ref,
                "obj": obj,
                "gap_pct": gap_pct,
                "feasible": r["feasible"],
                "timed_out": r["timed_out"],
                "time_s": r["time_s"],
                "items": r["total_items"],
                "aisles": r["num_aisles"],
                "orders": r["num_orders"],
                "n_orders": n_o,
                "n_items": n_i,
                "n_aisles": n_a,
            })

    rows.sort(key=lambda r: (r["dataset"], r["instance"]))

    # per-instance table
    print(
        f"{'ds':>2}  {'instance':<18} {'nOrd':>6} {'nItm':>6} {'nA':>4} | "
        f"{'best':>10} {'obj':>10} {'gap%':>7}  ord  ais  T(s)  flags"
    )
    print("-" * 110)
    for r in rows:
        flags = []
        if not r["feasible"]:
            flags.append("INFEASIBLE")
        if r["timed_out"]:
            flags.append("TO")
        if r["gap_pct"] is None or r["gap_pct"] != r["gap_pct"]:
            gap_str = "n/a"
        else:
            gap_str = f"{r['gap_pct']:>6.2f}"
        print(
            f"{r['dataset']:>2}  {r['instance']:<18} "
            f"{r['n_orders']:>6} {r['n_items']:>6} {r['n_aisles']:>4} | "
            f"{(r['ref'] or 0):>10.2f} {r['obj']:>10.2f} {gap_str}  "
            f"{r['orders']:>4} {r['aisles']:>3}  {r['time_s']:>5.1f}  "
            f"{','.join(flags)}"
        )

    # summary by dataset
    print()
    print("=== SUMMARY BY DATASET ===")
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for ds, rs in sorted(by_ds.items()):
        gaps = [r["gap_pct"] for r in rs if r["gap_pct"] == r["gap_pct"]]
        matched = sum(1 for g in gaps if g <= 0.001)
        within_1 = sum(1 for g in gaps if g <= 1.0)
        within_5 = sum(1 for g in gaps if g <= 5.0)
        timed_out = sum(1 for r in rs if r["timed_out"])
        infeas = sum(1 for r in rs if not r["feasible"])
        mean = sum(gaps) / len(gaps) if gaps else 0.0
        worst = max(gaps) if gaps else 0.0
        print(
            f"{ds}: n={len(rs)} matched={matched} "
            f"≤1%={within_1} ≤5%={within_5} mean_gap={mean:.2f}% "
            f"worst={worst:.2f}% timed_out={timed_out} infeasible={infeas}"
        )

    # worst gaps
    print()
    print("=== WORST 10 GAPS (any dataset) ===")
    worst_rows = sorted(
        [r for r in rows if r["gap_pct"] == r["gap_pct"]],
        key=lambda r: -r["gap_pct"],
    )[:10]
    print(
        f"{'ds':>2}  {'instance':<18} {'nOrd':>6} {'nA':>4} | "
        f"{'best':>10} {'obj':>10} {'gap%':>7}  ord  ais  TO"
    )
    for r in worst_rows:
        print(
            f"{r['dataset']:>2}  {r['instance']:<18} "
            f"{r['n_orders']:>6} {r['n_aisles']:>4} | "
            f"{(r['ref'] or 0):>10.2f} {r['obj']:>10.2f} "
            f"{r['gap_pct']:>6.2f}  {r['orders']:>4} {r['aisles']:>3}  "
            f"{'TO' if r['timed_out'] else ''}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python analyze_dalns.py <jsonl_path> [<jsonl_path> ...]")
        sys.exit(1)
    main(sys.argv[1:])
