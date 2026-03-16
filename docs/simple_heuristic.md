# SimpleHeuristic — Order Similarity Seed Algorithm

## Idea

Most heuristics for the Wave Order Picking problem start by selecting aisles and then fitting orders into them. **SimpleHeuristic** inverts this logic: it selects orders first, completely ignoring aisle information, and only resolves which aisles are needed once the wave is assembled.

The core intuition is that orders that share many of the same items are likely to be fulfilled by the same aisles. By grouping similar orders together, we expect to cover their demand with fewer aisle visits — maximising the objective `total_units / n_aisles_visited` — without ever looking at the aisle data until the last step.

---

## The Seed

A **seed** is an externally provided list of order indices that anchors the wave. These orders are added to the wave first (as long as they fit within the capacity upper bound `UB`). The union of all items appearing in the seed orders forms the **reference item set**, which is the "profile" used to score every other order.

The seed is intentionally given by the caller, not chosen internally, which makes the heuristic easy to embed in a larger meta-heuristic or parameter sweep that controls which seeds are tried.

---

## Similarity Metric

Two orders are compared using the **Jaccard index** on their item key sets (item presence, not quantities):

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where $A$ and $B$ are the sets of distinct item IDs required by each order.

- $J = 1.0$: both orders request exactly the same items.
- $J = 0.0$: the orders share no items, or one of them is empty.

The similarity function is isolated in `_similarity(items_a, items_b)` so it can be replaced with any other metric (e.g. cosine similarity on quantity vectors) without touching the rest of the algorithm.

---

## Algorithm

### Step-by-step

```
Input: orders, aisles, LB, UB, seed (list of order indices)

1. INIT WAVE
   For each order in seed (in order):
     If total_units + order_units <= UB:
       Add order to wave
       Add its items to ref_items

2. RANK REMAINING ORDERS
   For each order not in seed:
     sim = Jaccard(ref_items, order's item keys)
   Sort by sim descending

3. GREEDY FILL
   For each (sim, order) in sorted list:
     If total_units + order_units <= UB:
       Add order to wave

4. FEASIBILITY CHECK
   If total_units < LB:
     Return empty solution (wave is too small)

5. AISLE SELECTION (greedy)
   For each item demanded by the wave:
     While supply < demand:
       Add the first aisle (by index) that has stock of this item

6. Return (selected_orders, sorted(visited_aisles))
```

### Complexity

| Phase | Cost |
|---|---|
| Init wave | O(|seed| · items_per_order) |
| Rank orders | O(n_orders · items_per_order) |
| Sort | O(n_orders log n_orders) |
| Greedy fill | O(n_orders) |
| Aisle selection | O(n_items · n_aisles) |

---

## Worked Example

Using the unit-test instance from the problem description:

**Orders × Items** (`u_oi`):

| Order | Item 0 | Item 1 | Item 2 | Item 3 | Item 4 | Units |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 3 | 0 | 1 | 0 | 0 | **4** |
| 1 | 0 | 1 | 0 | 1 | 0 | **2** |
| 2 | 0 | 0 | 1 | 0 | 2 | **3** |
| 3 | 1 | 0 | 2 | 1 | 1 | **5** |
| 4 | 0 | 1 | 0 | 0 | 0 | **1** |

**Parameters:** `LB = 5`, `UB = 12`, `seed = [0]`

---

### Step 1 — Init wave from seed

Seed order 0 has 4 units. `4 ≤ 12` → add it.

```
selected_orders = [0]
total_units     = 4
ref_items       = {0, 2}          # items requested by order 0
```

### Step 2 — Compute similarities to `{0, 2}`

| Order | Items | Intersection | Union | Jaccard |
|:---:|:---:|:---:|:---:|:---:|
| 1 | {1, 3} | {} | {0, 1, 2, 3} | 0/4 = **0.00** |
| 2 | {2, 4} | {2} | {0, 2, 4} | 1/3 = **0.33** |
| 3 | {0, 2, 3, 4} | {0, 2} | {0, 2, 3, 4} | 2/4 = **0.50** |
| 4 | {1} | {} | {0, 1, 2} | 0/3 = **0.00** |

Sorted descending: **3 (0.50), 2 (0.33), 1 (0.00), 4 (0.00)**

### Step 3 — Greedy fill (UB = 12)

| Candidate | Sim | Units | total_units after | Fits? |
|:---:|:---:|:---:|:---:|:---:|
| 3 | 0.50 | 5 | 4+5=9 | ✓ add |
| 2 | 0.33 | 3 | 9+3=12 | ✓ add |
| 1 | 0.00 | 2 | 12+2=14 | ✗ skip (>12) |
| 4 | 0.00 | 1 | 12+1=13 | ✗ skip (>12) |

```
selected_orders = [0, 3, 2]
total_units     = 12
```

### Step 4 — LB check

`12 ≥ 5` → feasible, proceed.

### Step 5 — Aisle selection

**Aisles × Items** (`u_ai`):

| Aisle | Item 0 | Item 1 | Item 2 | Item 3 | Item 4 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 2 | 1 | 1 | 0 | 1 |
| 1 | 2 | 1 | 2 | 0 | 1 |
| 2 | 0 | 2 | 0 | 1 | 2 |
| 3 | 2 | 1 | 0 | 1 | 1 |
| 4 | 0 | 1 | 2 | 1 | 2 |

Aggregated demand from orders {0, 3, 2}:

| Item | Demand |
|:---:|:---:|
| 0 | 3+1 = **4** |
| 2 | 1+2+1 = **4** |
| 3 | 1 |
| 4 | 1+2 = **3** |

Greedy aisle selection per item:

- **Item 0**, need 4: aisle 0 gives 2 (supply=2 < 4) → add aisle 0. Still short → aisle 1 gives 2 more (supply=4 ≥ 4) → add aisle 1. ✓
- **Item 2**, need 4: aisle 1 already added, gives 2. Still short → aisle 4 gives 2 more (supply=4 ≥ 4) → add aisle 4. ✓
- **Item 3**, need 1: aisle 4 already added, gives 1 ≥ 1. ✓
- **Item 4**, need 3: aisle 0 gives 1, aisle 1 gives 1, aisle 4 gives 2 → total 4 ≥ 3. ✓

```
visited_aisles = [0, 1, 4]
```

### Result

```
selected_orders = [0, 3, 2]  → 12 total units
visited_aisles  = [0, 1, 4]  → 3 aisles
objective       = 12 / 3 = 4.0
```

(The optimal solution for this instance is 5.0, achieved by a different seed. This example shows a valid feasible wave.)

---

## Notes

- **No aisle awareness during order selection.** The heuristic deliberately avoids using aisle data when ranking orders. This makes it fast and keeps the order-selection and aisle-selection concerns separated.
- **Seed choice drives quality.** A seed that represents a "profitable" item profile will attract similar high-value orders. Trying multiple seeds (e.g., all orders, or the `k` largest) and keeping the best wave is a natural extension.
- **Aisle selection is not minimised.** The greedy aisle step adds aisles in index order, which is not guaranteed to minimise `n_aisles`. A set-cover or greedy-by-coverage approach could reduce this further.

---

## Implementation Insights & Bug Analysis

### Was the previous implementation correct?

No. The original `simple_heuristic.py` contained three categories of bugs that caused it to behave as a plain **First-Come-First-Served (FCFS)** greedy rather than the documented Jaccard-similarity-seed algorithm.

---

### Bug 1 — Jaccard ranking was never performed

The documented algorithm ranks *remaining* orders (those not in the seed) by Jaccard similarity to the seed's item profile, and then fills the wave in that order. The original code never computed similarities; it simply iterated through the seed list in order and stopped after exhausting it. Because `main.py` provided the **complete** set of orders (in a random permutation) as the seed, there were **zero remaining orders** to rank, and the similarity-ranking phase was completely bypassed.

**Impact:** The algorithm degraded to "shuffle all orders, greedily pick from front until UB is reached" — i.e., FCFS with a random shuffle.

---

### Bug 2 — No lower-bound (LB) feasibility check

The documented Step 4 rejects the wave if `total_units < LB` and returns an empty solution. The original code had no such guard, so it could emit solutions that violated the minimum wave size.

---

### Bug 3 — Broken aisle selection (loop-variable scope + no supply accumulation)

```python
# Original (buggy) aisle selection
for item_aisle in aisle.keys():
    if item_order != item_aisle:
        continue           # <-- only `continue`, the loop keeps going

if aisle[item_aisle] >= quantity_order:   # item_aisle = LAST key in dict, not item_order
    aisle[item_order] = aisle[item_aisle] - quantity_order
    selected_aisles.add(aisle_idx)        # always adds the aisle regardless of prior supply
```

Two problems here:

1. **Python loop variable retention after the loop exits:** after the `for item_aisle in aisle.keys()` loop completes, `item_aisle` holds the **last key** iterated over. Because the loop body only skips iterations with `continue` (rather than storing the matching key), there is no guarantee that `item_aisle == item_order` after the loop. The subsequent `aisle[item_aisle]` look-up therefore reads the wrong item.
2. **No supply accumulation:** the code checks `aisle[item_aisle] >= quantity_order` per-aisle in isolation. It adds an aisle whenever its individual stock meets the per-order demand, without accounting for how much supply has already been gathered from previously selected aisles. This causes both over-selection (adding redundant aisles) and under-selection (skipping aisles where partial supply is needed to satisfy cumulative demand).

---

### Why were results similar or identical?

Three compounding reasons:

1. **Seed = full permutation.** `main.py` built seeds as `random.sample(range(nOrders), nOrders)` — a complete random shuffle of every order. The documented algorithm's Jaccard-ranking phase therefore had *no* remaining orders to rank; every run was just FCFS with a different shuffle.

2. **FCFS converges to the same capacity.** A greedy selection that stops at `UB` tends to reach approximately the same total-units regardless of shuffle order (it keeps adding orders until it *can't* add another without exceeding UB). Different shuffles usually end up picking a similar or identical number of orders, yielding the same numerator `total_units` in the objective.

3. **Aisle-selection bug inflated denominator uniformly.** Because the buggy aisle loop added aisles indiscriminately, many runs produced the same inflated `n_aisles`, keeping the ratio `total_units / n_aisles` nearly constant across runs.

---

### Fixes applied

| Location | Change |
|---|---|
| `src/impl/simple_heuristic.py` | Full rewrite to implement the documented Jaccard-similarity-seed algorithm; added `_similarity()`, seed initialisation, similarity ranking of remaining orders, LB check, and correct per-item supply-accumulation aisle selection. |
| `src/main.py` | Changed seed generation from full random permutations to **single-order seeds**, so each of the 5 runs anchors the wave on a different order's item profile, and Jaccard ranking is actually exercised over the remaining orders. |
