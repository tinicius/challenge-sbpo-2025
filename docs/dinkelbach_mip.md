# Dinkelbach MIP — Implementation Explanation

## The Core Idea (Dinkelbach's Theorem)

Maximizing `f(x)/g(x)` is equivalent to finding `α*` such that:

```
max { f(x) - α* · g(x) } = 0
```

So instead of solving the fraction directly, you iterate:

1. Fix `α_k = items_k / aisles_k` (current ratio estimate)
2. Solve: **maximize** `items - α_k · aisles`
3. Update `α_{k+1}` with the new solution's ratio
4. Repeat until convergence

Each iteration's MIP is a standard linear 0-1 program — no fractions.

---

## Implementation Walkthrough

### `solve()` — main loop

1. **Preprocessing**: computes `order_sizes` and filters to `active_aisles` (only aisles that stock at least one demanded item — reduces MIP size).

2. **Warm-start** (`_warmstart()`): runs `AisleFirstHeuristic` to get an initial feasible solution and a non-trivial starting `α`. This matters because at `α=0` the MIP just maximizes items, which is far from the true optimum — wasting iteration budget.

3. **Dinkelbach loop** (up to `max_iters=15`):
   - Divides remaining time budget evenly across remaining iterations
   - Solves a MIP via CP-SAT
   - Checks convergence: stops if `|items - α·aisles| < ε` or ratio stopped improving

### `_solve_cp_sat()` — the parametric MIP

The objective is scaled to integers (CP-SAT requires integer coefficients):

```
maximize  q_d · Σ(s_o · y_o)  −  q_p · Σ(x_a)
```

where `q_d = n_aisles` and `q_p = items` from the previous iteration (so `q_p/q_d = α_k`).

**Constraints:**
- `LB ≤ Σ(s_o · y_o) ≤ UB` — wave size bounds
- `Σ demand(o,i)·y_o ≤ Σ supply(a,i)·x_a` for each item — stock coverage (can't pick an order if the visited aisles don't have enough stock)

**Hints**: Each iteration seeds CP-SAT with the previous solution via `AddHint`, so the solver starts from a known-feasible point and explores nearby.

---

## Convergence

| Condition | Meaning |
|---|---|
| `\|items - α·aisles\| < ε` | Dinkelbach residual is tiny — at optimum |
| `ratio ≤ α + 1e-9` | Ratio didn't improve — already optimal |
| Time budget exhausted | Falls back to best seen so far |

The best solution across all iterations is tracked and returned regardless of which iteration achieved it.

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `time_limit` | 590s | Total solver budget |
| `max_iters` | 15 | Max Dinkelbach iterations |
| `epsilon` | 1e-3 | Convergence threshold |
| `num_workers` | 8 | CP-SAT parallel workers |
