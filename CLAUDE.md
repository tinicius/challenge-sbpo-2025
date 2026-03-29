# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a solver suite for the **Mercado Libre First Optimization Challenge (SBPO 2025)** — a wave order-picking optimization problem. The goal is to maximize `total_items_collected / num_aisles_visited` by selecting a subset of orders and aisles that respects wave-size bounds (LB ≤ selected orders ≤ UB) and stock-availability constraints.

## Running

```bash
# Activate virtual environment
source venv/bin/activate

# Run benchmark on dataset A (default)
python src/main.py

# Validate a solution manually
python checker.py <instance_file> <solution_file>
```

Configuration is in `src/main.py` — edit `solver_configs`, `input_folder`, `RUNS`, and `SOLVER_TIME_LIMIT_SECONDS` to change which solvers run, on which dataset, and for how long.

## Dependencies

- Python 3.8+, numpy, pandas
- Unix-only: uses `SIGALRM` for solver timeouts

## Architecture

```
src/
├── main.py                   # Entry point: configures and runs benchmarks
├── models/solver.py          # ProblemInput dataclass + abstract Solver base class
├── utils/
│   ├── read_input.py         # Parses .txt instance files → ProblemInput
│   ├── generate_output.py    # Writes solution to .txt format
│   ├── wave_order_picking.py # is_solution_feasible() + compute_objective_function()
│   └── print_table.py        # Debug table printing
└── impl/                     # All solver implementations
    └── utils/                # Shared greedy building blocks
```

**Core data types** (`models/solver.py`):
- `ProblemInput`: `nOrders`, `nItems`, `nAisles`, `orders[o][item]=qty`, `aisles[a][item]=qty`, `lb`, `ub`
- `Solver` (ABC): `solve(problem: ProblemInput) -> tuple[list[int], list[int]]` — returns `(selected_orders, visited_aisles)`

**Objective:** `compute_objective_function()` in `utils/wave_order_picking.py` returns `total_items / num_aisles`. Feasibility is checked with `is_solution_feasible()`.

## Adding a New Solver

1. Create `src/impl/my_solver.py` extending `Solver` and implementing `solve()`
2. Register it in `src/main.py` inside `solver_configs` as a `RunConfig`

## Output

- Solutions: `output/` (one `.txt` per instance)
- Benchmark CSVs: `objectives_output/{solver_name}_{dataset}.csv` — columns: instance, feasible_runs, timeout_runs, obj_mean/median/min/max/variance, items_mean, aisles_mean, exec_time_mean
- `objectives/` — same format but written during runs (pre-aggregation)

## Datasets

- `datasets/a/` — 20 qualification instances
- `datasets/b/` — 15 sprint instances
- `datasets/x/` — 15 final instances
