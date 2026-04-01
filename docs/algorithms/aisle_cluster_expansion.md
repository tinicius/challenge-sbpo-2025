# AisleClusterExpansion

## What it is
AisleClusterExpansion is a seed-and-expand heuristic that starts from one aisle and iteratively adds aisles that best reduce missing demand from partially covered orders.

It is implemented in `algorithms/aisle_cluster_expansion.py` and registered as `aisle_cluster_expansion`.

## Why it is important
This heuristic is important because it models a practical warehouse intuition:

- begin with a local aisle choice
- observe what demand is still missing
- expand only with aisles that add the most useful missing stock

This targeted expansion avoids unnecessary aisles and can produce high item-per-aisle ratios.

## Core idea
The objective is:

- maximize total selected units / number of visited aisles

The algorithm samples seed aisles, grows each aisle cluster by demand deficits, and evaluates feasible order subsets when full order coverability becomes available.

## How it works
1. Compute order unit totals.
2. Randomly shuffle aisle seeds and keep a configured number of attempts.
3. For each seed aisle:
   - start with one-aisle cluster
   - build pooled inventory for current cluster
   - classify orders as untouched, partially touched, or fully coverable
4. When fully coverable orders exist:
   - pack a feasible subset under UB
   - test both packing preferences (large-first and small-first)
   - accept only subsets with total units >= LB
   - evaluate objective and update best solution with tie-breakers
5. If no valid stop yet, expand cluster:
   - compute missing demand from partial orders
   - score each available aisle by how much missing demand it can cover
   - add the best-scoring aisle
6. Stop expansion when:
   - maximum added aisles is reached
   - no missing demand remains
   - no aisle provides positive coverage gain

## Main strengths
- Expansion is guided by actual deficits, not only global frequency.
- Tests different packing preferences to reduce ordering bias.
- Configurable exploration breadth via attempts and max added aisles.
- Can find compact high-value aisle sets.

## Main limitations
- Uses random seed sampling, so results may vary between runs.
- Local expansion may get trapped in suboptimal clusters.
- Early stopping after first feasible subset for a seed can miss later improvements within that seed path.

## Practical impact in this project
AisleClusterExpansion contributes by adding a complementary search behavior compared with more static greedy methods:

- better local-demand adaptation
- useful diversification in multi-algorithm experiments
- potential quality gains on instances where demand deficits are highly structured across aisles

It is especially useful in benchmark portfolios where diversity of heuristic behavior increases the chance of strong best-of-run outcomes.
