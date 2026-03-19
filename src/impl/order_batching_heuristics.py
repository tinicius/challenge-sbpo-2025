"""
Order Batching Heuristics for the Wave Order Picking Problem (SBPO 2025)

This module implements several construction heuristics and hybrid meta-heuristics
for the wave order picking problem. The goal is to select a subset of orders
(the *wave*) and the minimum set of aisles needed to cover them, maximising:

    objective = total_units_picked / number_of_aisles_visited

subject to wave size bounds [LB, UB] and per-item stock constraints.

Heuristics implemented
----------------------
Construction heuristics:
  LargestFirstHeuristic  – greedy, orders sorted by total units (descending).
  GreedyRatioHeuristic   – greedy, always picks the order that maximises
                           the objective after addition (direct ratio greedy).

Hybrid (construction + local search):
  LocalSearchHeuristic   – greedy construction followed by local search
                           (add / remove / swap moves).
  GRASPHeuristic         – Greedy Randomised Adaptive Search Procedure:
                           randomised greedy construction + local search,
                           repeated for multiple iterations.
"""

import random

from impl.utils.greedy_aisle_select import greedy_aisle_select
from models.solver import Solver

# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def _compute_total_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
    """Aggregate per-item stock across all aisles."""
    stock: dict[int, int] = {}
    for aisle in aisles:
        for item, qty in aisle.items():
            stock[item] = stock.get(item, 0) + qty
    return stock


def _compute_demand(
    selected_orders: list[int],
    orders: list[dict[int, int]],
) -> dict[int, int]:
    """Sum per-item quantities demanded by all selected orders."""
    demand: dict[int, int] = {}
    for idx in selected_orders:
        for item, qty in orders[idx].items():
            demand[item] = demand.get(item, 0) + qty
    return demand


def _is_demand_feasible(
    demand: dict[int, int],
    total_stock: dict[int, int],
) -> bool:
    """Return True if total supply can meet demand for every item."""
    for item, qty in demand.items():
        if qty > total_stock.get(item, 0):
            return False
    return True


def _compute_objective_from_demand(
    demand: dict[int, int],
    aisles: list[dict[int, int]],
) -> tuple[float, list[int]]:
    """
    Given a demand dict, select the minimum aisles via the greedy heuristic
    and return (objective_value, visited_aisles).
    """
    visited = greedy_aisle_select(demand, aisles)
    if not visited:
        return 0.0, []
    return sum(demand.values()) / len(visited), visited


def _compute_objective(
    selected_orders: list[int],
    orders: list[dict[int, int]],
    aisles: list[dict[int, int]],
) -> tuple[float, list[int]]:
    """Compute objective value and visited aisles for a set of order indices."""
    if not selected_orders:
        return 0.0, []
    demand = _compute_demand(selected_orders, orders)
    return _compute_objective_from_demand(demand, aisles)


# ---------------------------------------------------------------------------
# Local search shared implementation
# ---------------------------------------------------------------------------


def _local_search(
    initial_orders: list[int],
    orders: list[dict[int, int]],
    aisles: list[dict[int, int]],
    total_stock: dict[int, int],
    lb: int,
    ub: int,
    n_orders: int,
    max_iter: int = 100,
) -> tuple[list[int], list[int]]:
    """
    Improve a feasible solution through add, remove, and swap moves.

    Strategy: first-improvement within each move category, tried in the
    order: Add → Remove → Swap.  Terminates when no improving move is found
    or *max_iter* iterations have been completed.

    Parameters
    ----------
    initial_orders : list[int]
        Starting set of order indices (must be feasible).
    orders, aisles : lists
        Problem data from the Solver.
    total_stock : dict[int, int]
        Aggregated per-item supply across all aisles.
    lb, ub : int
        Wave size bounds (total units).
    n_orders : int
        Total number of orders in the backlog.
    max_iter : int
        Maximum number of improvement iterations.

    Returns
    -------
    (improved_orders, visited_aisles)
    """
    current_orders = list(initial_orders)
    current_obj, current_aisles = _compute_objective(current_orders, orders, aisles)
    current_units = sum(sum(orders[i].values()) for i in current_orders)

    all_indices = set(range(n_orders))

    for _ in range(max_iter):
        improved = False
        current_set = set(current_orders)
        outside = list(all_indices - current_set)

        # ── Add move ──────────────────────────────────────────────────────
        for add_idx in outside:
            add_units = sum(orders[add_idx].values())
            if current_units + add_units > ub:
                continue
            new_orders = current_orders + [add_idx]
            new_demand = _compute_demand(new_orders, orders)
            if not _is_demand_feasible(new_demand, total_stock):
                continue
            new_obj, new_aisles = _compute_objective_from_demand(new_demand, aisles)
            if new_obj > current_obj:
                current_orders = new_orders
                current_obj = new_obj
                current_aisles = new_aisles
                current_units += add_units
                improved = True
                break

        if improved:
            continue

        # ── Remove move ───────────────────────────────────────────────────
        for i, rem_idx in enumerate(current_orders):
            rem_units = sum(orders[rem_idx].values())
            if current_units - rem_units < lb:
                continue
            new_orders = current_orders[:i] + current_orders[i + 1 :]
            if not new_orders:
                continue
            new_obj, new_aisles = _compute_objective(new_orders, orders, aisles)
            if new_obj > current_obj:
                current_orders = new_orders
                current_obj = new_obj
                current_aisles = new_aisles
                current_units -= rem_units
                improved = True
                break

        if improved:
            continue

        # ── Swap move ─────────────────────────────────────────────────────
        current_set = set(current_orders)
        outside = list(all_indices - current_set)
        swapped = False
        for i, rem_idx in enumerate(current_orders):
            rem_units = sum(orders[rem_idx].values())
            for add_idx in outside:
                add_units = sum(orders[add_idx].values())
                new_units = current_units - rem_units + add_units
                if not (lb <= new_units <= ub):
                    continue
                new_orders = current_orders[:i] + [add_idx] + current_orders[i + 1 :]
                new_demand = _compute_demand(new_orders, orders)
                if not _is_demand_feasible(new_demand, total_stock):
                    continue
                new_obj, new_aisles = _compute_objective_from_demand(
                    new_demand, aisles
                )
                if new_obj > current_obj:
                    current_orders = new_orders
                    current_obj = new_obj
                    current_aisles = new_aisles
                    current_units = new_units
                    improved = True
                    swapped = True
                    break
            if swapped:
                break

        if not improved:
            break

    return current_orders, current_aisles


# ---------------------------------------------------------------------------
# Heuristic classes
# ---------------------------------------------------------------------------


class LargestFirstHeuristic(Solver):
    """
    Sorts orders by total units (largest first) and greedily adds them to
    the wave.  High-volume orders contribute many units per pick, which
    tends to produce a good units/aisles ratio quickly.

    Config keys
    -----------
    seed : list[int], optional
        Subset of order indices to consider.  Defaults to all orders.
    """

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self.config.get("seed", list(range(self.n_orders)))

        total_stock = _compute_total_stock(self.aisles)
        remaining_stock = dict(total_stock)

        sorted_orders = sorted(
            seed,
            key=lambda idx: sum(self.orders[idx].values()),
            reverse=True,
        )

        selected_orders: list[int] = []
        total_units = 0

        for order_idx in sorted_orders:
            order = self.orders[order_idx]
            if any(remaining_stock.get(item, 0) < qty for item, qty in order.items()):
                continue
            order_units = sum(order.values())
            if total_units + order_units > self.ub:
                continue
            selected_orders.append(order_idx)
            total_units += order_units
            for item, qty in order.items():
                remaining_stock[item] -= qty

        if total_units < self.lb:
            return [], []

        demand = _compute_demand(selected_orders, self.orders)
        visited_aisles = greedy_aisle_select(demand, self.aisles)
        return selected_orders, visited_aisles


class GreedyRatioHeuristic(Solver):
    """
    At each construction step, evaluates every candidate order and selects
    the one whose addition yields the highest units/aisles objective value.
    This directly optimises the fractional objective during construction.

    Note: This heuristic performs O(n²) calls to *greedy_aisle_select* and
    may be slow on very large instances.

    Config keys
    -----------
    seed : list[int], optional
        Subset of order indices to consider.  Defaults to all orders.
    """

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self.config.get("seed", list(range(self.n_orders)))

        total_stock = _compute_total_stock(self.aisles)
        remaining_stock = dict(total_stock)

        selected_orders: list[int] = []
        total_units = 0
        remaining = list(seed)

        while remaining:
            best_score = -1.0
            best_pos = -1
            best_order_idx = -1

            for pos, order_idx in enumerate(remaining):
                order = self.orders[order_idx]
                if any(
                    remaining_stock.get(item, 0) < qty for item, qty in order.items()
                ):
                    continue
                order_units = sum(order.values())
                if total_units + order_units > self.ub:
                    continue
                score, _ = _compute_objective(
                    selected_orders + [order_idx], self.orders, self.aisles
                )
                if score > best_score:
                    best_score = score
                    best_pos = pos
                    best_order_idx = order_idx

            if best_pos == -1:
                break

            order_units = sum(self.orders[best_order_idx].values())
            selected_orders.append(best_order_idx)
            total_units += order_units
            remaining.pop(best_pos)
            for item, qty in self.orders[best_order_idx].items():
                remaining_stock[item] -= qty

        if total_units < self.lb:
            return [], []

        _, visited_aisles = _compute_objective(
            selected_orders, self.orders, self.aisles
        )
        return selected_orders, visited_aisles


class LocalSearchHeuristic(Solver):
    """
    Hybrid heuristic: greedy construction followed by local search.

    An initial solution is built with a simple greedy method, then
    iteratively improved by add, remove, and swap order moves that
    increase the units/aisles objective.

    Config keys
    -----------
    seed : list[int], optional
        Order in which to consider orders during construction.
        Defaults to all orders in their natural index order.
    construction : str, optional
        ``'simple'`` – follow *seed* order (default).
        ``'largest'`` – sort by order size before greedy pass.
    max_iter : int, optional
        Maximum local-search improvement iterations.  Default: 100.
    """

    def _construct(
        self,
        seed: list[int],
        remaining_stock: dict[int, int],
        largest_first: bool = False,
    ) -> tuple[list[int], int]:
        order_list = (
            sorted(seed, key=lambda idx: sum(self.orders[idx].values()), reverse=True)
            if largest_first
            else seed
        )
        selected: list[int] = []
        total = 0
        for order_idx in order_list:
            order = self.orders[order_idx]
            if any(remaining_stock.get(item, 0) < qty for item, qty in order.items()):
                continue
            units = sum(order.values())
            if total + units > self.ub:
                continue
            selected.append(order_idx)
            total += units
            for item, qty in order.items():
                remaining_stock[item] -= qty
        return selected, total

    def solve(self) -> tuple[list[int], list[int]]:
        seed = self.config.get("seed", list(range(self.n_orders)))
        construction = self.config.get("construction", "simple")
        max_iter = self.config.get("max_iter", 100)

        total_stock = _compute_total_stock(self.aisles)
        remaining_stock = dict(total_stock)

        initial_orders, total_units = self._construct(
            seed, remaining_stock, largest_first=(construction == "largest")
        )

        if total_units < self.lb:
            return [], []

        result_orders, result_aisles = _local_search(
            initial_orders,
            self.orders,
            self.aisles,
            total_stock,
            self.lb,
            self.ub,
            self.n_orders,
            max_iter,
        )

        return result_orders, result_aisles


class GRASPHeuristic(Solver):
    """
    GRASP (Greedy Randomised Adaptive Search Procedure) heuristic.

    Each iteration:
    1. **Randomised greedy construction** – at every step, compute a score
       for each feasible candidate order, build a *Restricted Candidate
       List* (RCL) of the top-*alpha* fraction, and choose uniformly at
       random from the RCL.
    2. **Local search** – improve the constructed solution via add / remove
       / swap moves.
    3. **Update incumbent** – keep the best solution found across iterations.

    Config keys
    -----------
    n_iter : int, optional
        Number of GRASP iterations.  Default: 10.
    alpha : float, optional
        RCL size as a fraction of feasible candidates (0 = fully greedy,
        1 = fully random).  Default: 0.3.
    max_ls_iter : int, optional
        Maximum local-search improvement iterations per GRASP round.
        Default: 50.
    """

    def _grasp_construct(
        self,
        seed: list[int],
        total_stock: dict[int, int],
        alpha: float,
    ) -> list[int]:
        """Randomised greedy construction phase of one GRASP iteration."""
        remaining_stock = dict(total_stock)
        selected_orders: list[int] = []
        total_units = 0
        remaining = list(seed)

        while remaining:
            # Score every feasible candidate
            candidates: list[tuple[float, int, int]] = []
            for pos, order_idx in enumerate(remaining):
                order = self.orders[order_idx]
                if any(
                    remaining_stock.get(item, 0) < qty for item, qty in order.items()
                ):
                    continue
                order_units = sum(order.values())
                if total_units + order_units > self.ub:
                    continue
                score, _ = _compute_objective(
                    selected_orders + [order_idx], self.orders, self.aisles
                )
                candidates.append((score, pos, order_idx))

            if not candidates:
                break

            # Build RCL: top-alpha fraction ranked by score
            candidates.sort(key=lambda x: x[0], reverse=True)
            rcl_size = max(1, int(len(candidates) * alpha))
            rcl = candidates[:rcl_size]

            _, chosen_pos, chosen_idx = random.choice(rcl)

            order_units = sum(self.orders[chosen_idx].values())
            selected_orders.append(chosen_idx)
            total_units += order_units
            remaining.pop(chosen_pos)
            for item, qty in self.orders[chosen_idx].items():
                remaining_stock[item] -= qty

        return selected_orders if total_units >= self.lb else []

    def solve(self) -> tuple[list[int], list[int]]:
        n_iter = self.config.get("n_iter", 10)
        alpha = self.config.get("alpha", 0.3)
        max_ls_iter = self.config.get("max_ls_iter", 50)

        total_stock = _compute_total_stock(self.aisles)
        seed = list(range(self.n_orders))

        best_orders: list[int] = []
        best_aisles: list[int] = []
        best_obj = -1.0

        for _ in range(n_iter):
            random.shuffle(seed)

            # Construction phase
            initial = self._grasp_construct(seed, total_stock, alpha)
            if not initial:
                continue

            # Local search improvement phase
            improved_orders, improved_aisles = _local_search(
                initial,
                self.orders,
                self.aisles,
                total_stock,
                self.lb,
                self.ub,
                self.n_orders,
                max_ls_iter,
            )

            if not improved_orders:
                continue

            obj, _ = _compute_objective(improved_orders, self.orders, self.aisles)
            if obj > best_obj:
                best_obj = obj
                best_orders = improved_orders
                best_aisles = improved_aisles

        return best_orders, best_aisles
