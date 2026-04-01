# AisleFirstHeuristic

## What it is
AisleFirstHeuristic is a greedy constructive algorithm that starts from promising aisles and then packs as many feasible orders as possible within the wave limits.

It is implemented in `algorithms/aisle_first.py` and registered as `aisle_first`.

## Why it is important
This heuristic is important because it quickly builds high-quality feasible waves by focusing on aisle productivity first:

- It prioritizes aisles with high useful inventory for current global demand.
- It tries different order packing strategies (large orders first and small orders first).
- It cleans redundant aisles to improve the objective denominator.
- It is deterministic enough to be reproducible with fixed inputs while still exploring multiple combinations.

In practice, it is a strong baseline: fast, simple, and often competitive on objective value.

## Core idea
The objective is:

- maximize total selected units / number of visited aisles

So this method first chooses aisle subsets likely to cover many demanded items, then greedily packs feasible orders under stock and upper-bound constraints.

## How it works
1. Build total demand from all orders.
2. Score each aisle by useful inventory (inventory that matches demand).
3. Select top seed aisles (capped by a configurable limit and by a hard constant of 5).
4. For each seed aisle:
   - rank similar aisles by Jaccard similarity of item sets
   - build aisle subsets of size k (from larger to smaller)
5. For each aisle subset:
   - pool inventory
   - pack orders with one or more order sequences (desc/asc/seed)
   - keep only solutions with total units >= LB
   - remove redundant aisles while preserving demand coverability
6. Keep the best candidate by tie-breakers:
   - higher objective
   - if equal objective: more units
   - if equal units: fewer aisles

## Main strengths
- Fast for large instance sets.
- Strong objective focus due to aisle cleanup.
- Uses similarity to keep aisle combinations coherent.
- Supports configuration for search behavior (order strategy, seed aisles).

## Main limitations
- Greedy packing can miss globally better combinations.
- Aisle similarity is based only on item presence, not quantity magnitude.
- Exploration depth is intentionally bounded for speed.

## Practical impact in this project
AisleFirstHeuristic helps the benchmark suite by providing:

- a robust and efficient heuristic reference
- a method that balances quality and runtime
- useful comparative behavior versus LP-based and set-cover style methods

This makes it valuable both for production-like runs and as a baseline for evaluating newer algorithms.
