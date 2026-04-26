# Pan & Liu (1995) — SBPO adaptation

Heuristic family adapted from:

> Pan C-H, Liu S-Y. **A Comparative Study of Order Batching Algorithms.**
> *Omega, Int. J. Mgmt Sci.*, 23(6):691–700, 1995.

The original paper studies order batching in a man-on-board AS/RS and compares
16 algorithms built from 4 seed-selection rules (SD1–SD4) and 4 order-addition
rules (OA1–OA4), plus the SL (small/large) algorithm of Elsayed & Unal (1989).
They conclude that **SD3 + OA3** (economic convex hull area + similarity
coefficient) is the overall winner across shape factors, S/R capacities and
storage-assignment policies, and that **SL** is competitive when the S/R
capacity is small.

## Mapping AS/RS → SBPO

| AS/RS concept (Pan & Liu)                          | SBPO wave-order-picking analog                               |
| -------------------------------------------------- | ------------------------------------------------------------ |
| One batch per S/R tour                             | **One wave**, bounded by `LB ≤ total_units ≤ UB`             |
| S/R travel time (minimise)                         | **Number of visited aisles** (minimise for fixed units)      |
| Item (x, y) location                               | **Aisle indices** that carry the item                        |
| Economic Convex Hull (ECH) $E_i$ of order $i$      | **Aisle footprint** $F_i$ = aisles touching any item of $i$  |
| Area($E_i$)                                        | $|F_i|$                                                      |
| Similarity coef. $SC = |E_s \cap E_i| / |E_s \cup E_i|$ | **Jaccard**($F_{\text{batch}}, F_i$)                     |
| S/R capacity $C$                                   | Wave upper bound **UB**                                      |
| Cumulative seeding                                 | Batch footprint grows: $F_{\text{batch}} = \bigcup F_i$      |
| 6-D SFC $\theta$ value                             | Normalised footprint centroid: $\bar{a}_i / n_{\text{aisles}}$ |

The footprint $F_i$ is the SBPO counterpart to an order's "reachable region"
in the rack: the aisles that materially participate in fulfilling the order.
Jaccard of footprints measures geometric overlap of two orders in the sense
that matters for the objective — sharing aisles.

SD4 / OA4 are less well motivated in SBPO because the rack has no true 2-D
layout; a 1-D footprint centroid is used as a coarse positional proxy and is
included mainly for completeness of the grid.

## Rules

### Seed selection

| Rule | Picks the eligible order with …                                        |
| ---- | ---------------------------------------------------------------------- |
| SD1  | largest total quantity `sum(qty)` (paper: "largest number of items")   |
| SD2  | largest number of distinct items (paper: "greatest total weight")      |
| SD3  | largest aisle footprint $|F_i|$ — the SBPO ECH analog                  |
| SD4  | smallest footprint centroid $\theta_i$ (closest to the "origin")       |

An order is *eligible* iff `0 < size ≤ UB` and aggregate stock covers it.

### Order addition (with **cumulative seeding**)

Let $F_B$ be the batch's current footprint and $F_i$ the candidate's.
| Rule | Selection |
|------|-----------|
| OA1  | $\max \|F_i \cap F_B\|$ — most shared aisles with the seed |
| OA2  | $\min \|F_i \setminus F_B\|$ — fewest aisles to *add* to the batch |
| OA3  | $\max \|F_i \cap F_B\| / \|F_i \cup F_B\|$ — Jaccard similarity (paper's SC) |
| OA4  | $\min \|\theta_i - \bar{\theta}_B\|$ — closest positional value |
Ties broken by largest order size so the wave fills faster.

### SL (small / large)

- Classify orders: **large** if `size ≥ sl_threshold · UB`, else **small**.
- Seed: the pair of **large** orders with maximum saving
  $|F_a \cap F_b|$ that still fits jointly in UB and stock. Falls back to the
  single largest-footprint large order, or the biggest small order if no
  large exists.
- Phase 1: greedily add remaining **large** orders by max saving against the
  growing batch footprint.
- Phase 2: same with **small** orders.

The "saving" is the SBPO equivalent of the travel-time saving in the EQUAL /
Clarke-Wright base of Elsayed & Unal's SL: shared aisles are "aisles you do
not pay again for".

## Parameters

```yaml
params:
  seed_rule: SD3     # SD1 | SD2 | SD3 | SD4 | SL
  add_rule:  OA3     # OA1 | OA2 | OA3 | OA4 (ignored when seed_rule=SL)
  greedy:    simple  # simple | multi — for the final min-aisle cover
  sl_threshold: 0.5  # only for seed_rule=SL; fraction of UB
```

## Recommended configuration

`pan_liu_sd3_oa3` — the winner reported by Pan & Liu (1995). For small-wave
instances (tight UB relative to order sizes) `pan_liu_sl` is also
competitive, per the paper's small-capacity analysis.

## Files

- `pan_liu_heuristic.py` — `PanLiuHeuristic` class (registered as `pan_liu`)
- Config: `configs/pan_liu.yaml` (16 SD×OA cells + SL variants)
