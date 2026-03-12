# Set Cover with Quantities — Capacitated Covering

This transforms the problem from a pure coverage problem into a **multi-commodity demand satisfaction** problem.

---

## New Problem Definition

Given:
- Universe $U = \{e_1, \ldots, e_n\}$ — items to pick
- **Demand** $d_j \geq 1$ — quantity of item $j$ required (sum across selected orders in the wave)
- Each aisle $i$ has **supply** $q_{ij} \geq 0$ — available quantity of item $j$ in aisle $i$
- **Goal:** Select a set of aisles $C$ and, for each item $j$, decide how much to take from each selected aisle, such that all demands are met, minimizing $|C|$

---

## Why It's Fundamentally Different

In the classic Set Cover, visiting an aisle either covers an item or it doesn't (binary). Now:

- An item $j$ with demand $d_j = 5$ might need **two aisles** even if both stock it, because each only has 3 units
- Conversely, a single aisle with large supply may satisfy demand alone
- You must also decide **how much to take from each aisle** (allocation subproblem)

---

## ILP Formulation

Let:
- $x_i \in \{0,1\}$ — whether aisle $i$ is visited
- $y_{ij} \geq 0$ — quantity of item $j$ taken from aisle $i$

$$\min \sum_{i} x_i$$

$$\text{s.t.}$$

$$\sum_{i} y_{ij} \geq d_j \qquad \forall j \in U \quad \text{(demand satisfied)}$$

$$y_{ij} \leq q_{ij} \cdot x_i \qquad \forall i, j \quad \text{(can only take from visited aisle, up to supply)}$$

$$y_{ij} \geq 0, \quad x_i \in \{0,1\}$$

The constraint $y_{ij} \leq q_{ij} \cdot x_i$ is a **big-M style coupling** — if $x_i = 0$, no item can come from aisle $i$.

---

## Key Difference from Classic Set Cover

| Classic | With Quantities |
|---|---|
| $A_{ji} \in \{0,1\}$: item $j$ present in aisle $i$ | $q_{ij} \geq 0$: supply of item $j$ in aisle $i$ |
| Demand always 1 | Demand $d_j \geq 1$ |
| Visiting an aisle = fully covers all its items | Visiting an aisle = access to its supply, may not be enough alone |
| Pure covering constraint | Covering constraint + flow/allocation |
| **NP-hard** | **Harder** — even the LP relaxation is more complex |

---

## Modified Greedy

The greedy rule must now account for quantities. One natural adaptation:

```
unsatisfied[j] = d[j]  for all j
C = ∅

while any unsatisfied[j] > 0:
    # score each aisle by how much demand it can fulfill
    for each aisle i not in C:
        score(i) = number of items j where min(q[i][j], unsatisfied[j]) > 0
        # or: score(i) = sum_j min(q[i][j], unsatisfied[j])  (quantity-weighted)

    i* = argmax score(i)
    C ← C ∪ {i*}

    for each item j:
        taken = min(q[i*][j], unsatisfied[j])
        unsatisfied[j] -= taken
```

Two scoring options:
- **Item-count score**: counts how many distinct items get at least partial satisfaction — mimics classic greedy
- **Quantity score**: $\sum_j \min(q_{ij}, \text{unsatisfied}_j)$ — greedy on total units fulfilled

The quantity-weighted score tends to be better in practice because it accounts for the magnitude of contribution.

---

## LP Relaxation Bound (Lower Bound)

Relax $x_i \in [0,1]$. The LP is now a **covering LP with coupling constraints** — solvable in polynomial time, and its optimal value is a valid lower bound on the ILP optimum. Useful for:

1. Bounding the optimality gap of heuristic solutions
2. Warm-starting the ILP solver

---

## Additional Complexity: Infeasibility

It's now possible that **no feasible solution exists** if for some item $j$:

$$\sum_{i} q_{ij} < d_j$$

i.e., total supply across all aisles cannot meet demand. In the warehouse context this would indicate a data inconsistency (an order was accepted that can't be fulfilled), but it's worth checking:

```python
for j in items:
    total_supply = sum(aisle[j] for aisle in aisles if j in aisle)
    assert total_supply >= demand[j], f"Item {j} infeasible"
```

---

## Connection Back to Your Data

In `read_input`, the structure already supports this:
- `aisles[i]` is a `dict` mapping `item_idx → quantity` — this is exactly $q_{ij}$
- The **demand** $d_j$ comes from summing quantities across all orders selected in a wave:

$$d_j = \sum_{o \in \text{wave}} \text{orders}[o][j]$$

So the full pipeline per wave is:

1. Determine which orders are in the wave → compute $d_j$ per item
2. Solve the capacitated set cover to find minimum aisles $C$
3. Compute allocation $y_{ij}$ (which is just a bookkeeping step once $C$ is known — a simple greedy suffices)
