"""Pan & Liu (1995) order-batching heuristics adapted for the SBPO 2025 wave
order-picking problem.

Reference:
    Pan C-H, Liu S-Y (1995). "A Comparative Study of Order Batching Algorithms."
    Omega, Int. J. Mgmt Sci., 23(6), 691-700.

Original context (AS/RS, man-on-board S/R machine):
    The paper compares 16 algorithms built by combining 4 seed-selection rules
    (SD1..SD4) with 4 order-addition rules (OA1..OA4), plus the SL (small /
    large) algorithm. Objective is to minimise S/R travel time. Authors
    conclude that SD3 + OA3 (economic convex hull + similarity coefficient) is
    the overall winner.

SBPO adaptation:
    Batch -> wave (a single batch, bounded by LB <= sum(order sizes) <= UB).
    Travel time -> number of visited aisles (objective is total_items / aisles).
    Item location (x, y) -> aisle indices that contain the item.
    Economic Convex Hull (ECH) E_i of order i -> *aisle footprint* F_i: the
        set of aisles that contain at least one item of order i.
    Area(E_i) -> |F_i|.
    SC = area(E_s & E_i) / area(E_s | E_i) -> Jaccard(F_batch, F_i).
    S/R capacity C -> wave upper bound UB.
    Cumulative seeding rule: F_batch grows as orders are added.
    6-D SFC theta -> mean(aisle indices in F_i) / nAisles. A coarse 1-D
        positional proxy (there is no true 2-D warehouse layout in SBPO).
"""

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


_VALID_SEED_RULE = {"SD1", "SD2", "SD3", "SD4", "SL"}
_VALID_ADD_RULE = {"OA1", "OA2", "OA3", "OA4"}
_VALID_GREEDY = {"simple", "multi"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}


class PanLiuHeuristic(Algorithm):
    """Configurable Pan & Liu (1995) batching heuristic for SBPO."""

    def __init__(self, params: dict):
        super().__init__(params)

        seed_rule = params.get("seed_rule", "SD3")
        if seed_rule not in _VALID_SEED_RULE:
            raise ValueError(
                f"PanLiuHeuristic: invalid 'seed_rule'={seed_rule!r}; "
                f"expected one of {sorted(_VALID_SEED_RULE)}"
            )
        add_rule = params.get("add_rule", "OA3")
        if add_rule not in _VALID_ADD_RULE:
            raise ValueError(
                f"PanLiuHeuristic: invalid 'add_rule'={add_rule!r}; "
                f"expected one of {sorted(_VALID_ADD_RULE)}"
            )
        greedy = params.get("greedy", "simple")
        if greedy not in _VALID_GREEDY:
            raise ValueError(
                f"PanLiuHeuristic: invalid 'greedy'={greedy!r}; "
                f"expected one of {sorted(_VALID_GREEDY)}"
            )
        sl_threshold = params.get("sl_threshold", 0.5)
        try:
            sl_threshold = float(sl_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "PanLiuHeuristic: 'sl_threshold' must be a number"
            ) from exc
        if not 0.0 < sl_threshold <= 1.0:
            raise ValueError(
                "PanLiuHeuristic: 'sl_threshold' must lie in (0, 1]"
            )

        self._seed_rule = seed_rule
        self._add_rule = add_rule
        self._greedy = greedy
        self._sl_threshold = sl_threshold

    @property
    def name(self) -> str:
        return "pan_liu_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders
        n_aisles = instance.nAisles

        if n_orders == 0 or n_aisles == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in orders]
        stock = self._aggregate_stock(aisles)
        footprints = self._build_footprints(orders, aisles)
        theta = self._theta_values(footprints, n_aisles)

        if self._seed_rule == "SL":
            selected, demand, total_units = self._sl_construct(
                orders, footprints, order_sizes, stock, ub
            )
        else:
            selected, demand, total_units = self._sd_oa_construct(
                orders, footprints, theta, order_sizes, stock, ub
            )

        if not selected or total_units < lb:
            return dict(_EMPTY_RESULT)

        visited = (
            multi_greedy_aisle_select(demand, aisles)
            if self._greedy == "multi"
            else greedy_aisle_select(dict(demand), aisles)
        )
        if not visited:
            return dict(_EMPTY_RESULT)

        return {
            "selected_orders": selected,
            "visited_aisles": visited,
            "objective": total_units / len(visited),
        }

    # ------------------------------------------------------------------
    # SD* + OA* construction (cumulative seeding)
    # ------------------------------------------------------------------
    def _sd_oa_construct(
        self,
        orders: list[dict[int, int]],
        footprints: list[set[int]],
        theta: list[float],
        order_sizes: list[int],
        stock: dict[int, int],
        ub: int,
    ) -> tuple[list[int], dict[int, int], int]:
        seed_idx = self._pick_seed(orders, footprints, theta, order_sizes, stock, ub)
        if seed_idx is None:
            return [], {}, 0

        selected = [seed_idx]
        demand = dict(orders[seed_idx])
        total_units = order_sizes[seed_idx]
        stock_remaining = dict(stock)
        for it, q in orders[seed_idx].items():
            stock_remaining[it] -= q

        batch_fp = set(footprints[seed_idx])
        theta_sum = theta[seed_idx]
        theta_count = 1

        remaining = set(range(len(orders))) - {seed_idx}

        while remaining:
            candidates = [
                i for i in remaining
                if total_units + order_sizes[i] <= ub
                and all(
                    stock_remaining.get(it, 0) >= q
                    for it, q in orders[i].items()
                )
            ]
            if not candidates:
                break

            batch_theta = theta_sum / theta_count
            best = self._pick_addition(
                candidates, footprints, theta, batch_fp, batch_theta, order_sizes
            )

            selected.append(best)
            total_units += order_sizes[best]
            for it, q in orders[best].items():
                demand[it] = demand.get(it, 0) + q
                stock_remaining[it] -= q
            batch_fp |= footprints[best]  # cumulative seeding
            theta_sum += theta[best]
            theta_count += 1
            remaining.remove(best)

        return selected, demand, total_units

    def _pick_seed(
        self,
        orders: list[dict[int, int]],
        footprints: list[set[int]],
        theta: list[float],
        order_sizes: list[int],
        stock: dict[int, int],
        ub: int,
    ) -> int | None:
        eligible = [
            i for i in range(len(orders))
            if 0 < order_sizes[i] <= ub
            and all(stock.get(it, 0) >= q for it, q in orders[i].items())
        ]
        if not eligible:
            return None

        rule = self._seed_rule
        if rule == "SD1":
            # Largest number of items (total quantity).
            return max(eligible, key=lambda i: (order_sizes[i], len(orders[i])))
        if rule == "SD2":
            # Greatest "weight" interpreted as number of distinct items.
            return max(eligible, key=lambda i: (len(orders[i]), order_sizes[i]))
        if rule == "SD3":
            # Largest economic convex hull -> largest aisle footprint.
            return max(eligible, key=lambda i: (len(footprints[i]), order_sizes[i]))
        if rule == "SD4":
            # Smallest 6-D SFC theta -> smallest normalised footprint centroid.
            return min(eligible, key=lambda i: (theta[i], -order_sizes[i]))
        raise AssertionError(rule)

    def _pick_addition(
        self,
        candidates: list[int],
        footprints: list[set[int]],
        theta: list[float],
        batch_fp: set[int],
        batch_theta: float,
        order_sizes: list[int],
    ) -> int:
        rule = self._add_rule
        if rule == "OA1":
            # Largest number of common "locations" (aisles) with the seed.
            return max(
                candidates,
                key=lambda i: (len(footprints[i] & batch_fp), order_sizes[i]),
            )
        if rule == "OA2":
            # Minimum total "distance" -> fewest new aisles beyond batch footprint.
            return min(
                candidates,
                key=lambda i: (len(footprints[i] - batch_fp), -order_sizes[i]),
            )
        if rule == "OA3":
            # Largest SC = Jaccard(batch footprint, candidate footprint).
            def jaccard(i: int) -> float:
                fp = footprints[i]
                union = len(fp | batch_fp)
                return len(fp & batch_fp) / union if union else 0.0

            return max(candidates, key=lambda i: (jaccard(i), order_sizes[i]))
        if rule == "OA4":
            # Smallest deviation from batch's mean theta.
            return min(
                candidates,
                key=lambda i: (abs(theta[i] - batch_theta), -order_sizes[i]),
            )
        raise AssertionError(rule)

    # ------------------------------------------------------------------
    # SL (small / large) algorithm [Elsayed & Unal 1989]
    # ------------------------------------------------------------------
    def _sl_construct(
        self,
        orders: list[dict[int, int]],
        footprints: list[set[int]],
        order_sizes: list[int],
        stock: dict[int, int],
        ub: int,
    ) -> tuple[list[int], dict[int, int], int]:
        threshold = self._sl_threshold * ub
        eligible = [
            i for i in range(len(orders))
            if 0 < order_sizes[i] <= ub
            and all(stock.get(it, 0) >= q for it, q in orders[i].items())
        ]
        large = [i for i in eligible if order_sizes[i] >= threshold]
        small = [i for i in eligible if order_sizes[i] < threshold]

        selected: list[int] = []
        demand: dict[int, int] = {}
        total_units = 0
        stock_remaining = dict(stock)
        batch_fp: set[int] = set()

        def add(idx: int) -> None:
            nonlocal total_units
            selected.append(idx)
            total_units += order_sizes[idx]
            for it, q in orders[idx].items():
                demand[it] = demand.get(it, 0) + q
                stock_remaining[it] -= q
            batch_fp.update(footprints[idx])

        def feasible(idx: int) -> bool:
            if total_units + order_sizes[idx] > ub:
                return False
            return all(
                stock_remaining.get(it, 0) >= q for it, q in orders[idx].items()
            )

        # Seed: the pair of LARGE orders with the biggest travel-time saving,
        # i.e. largest |F_a & F_b|. Fall back to the single largest footprint
        # if no compatible pair exists, and finally to the biggest small order.
        seed_pair = self._best_large_pair(large, orders, footprints, order_sizes, stock, ub)
        if seed_pair is not None:
            add(seed_pair[0])
            add(seed_pair[1])
        elif large:
            add(max(large, key=lambda i: (len(footprints[i]), order_sizes[i])))
        elif small:
            add(max(small, key=order_sizes.__getitem__))

        # Phase 1: add remaining large orders by maximum saving.
        self._fill_by_saving(
            [i for i in large if i not in selected], footprints, batch_fp,
            order_sizes, feasible, add,
        )
        # Phase 2: add small orders by maximum saving.
        self._fill_by_saving(
            [i for i in small if i not in selected], footprints, batch_fp,
            order_sizes, feasible, add,
        )

        return selected, demand, total_units

    @staticmethod
    def _best_large_pair(
        large: list[int],
        orders: list[dict[int, int]],
        footprints: list[set[int]],
        order_sizes: list[int],
        stock: dict[int, int],
        ub: int,
    ) -> tuple[int, int] | None:
        best: tuple[int, int] | None = None
        best_saving = -1
        for x in range(len(large)):
            a = large[x]
            fp_a = footprints[a]
            order_a = orders[a]
            size_a = order_sizes[a]
            for y in range(x + 1, len(large)):
                b = large[y]
                if size_a + order_sizes[b] > ub:
                    continue
                order_b = orders[b]
                ok = True
                for it, q in order_a.items():
                    if stock.get(it, 0) < q + order_b.get(it, 0):
                        ok = False
                        break
                if not ok:
                    continue
                saving = len(fp_a & footprints[b])
                if saving > best_saving:
                    best_saving = saving
                    best = (a, b)
        return best

    @staticmethod
    def _fill_by_saving(
        pool: list[int],
        footprints: list[set[int]],
        batch_fp: set[int],
        order_sizes: list[int],
        feasible,
        add,
    ) -> None:
        pool = list(pool)
        while pool:
            cands = [i for i in pool if feasible(i)]
            if not cands:
                break
            best = max(
                cands,
                key=lambda i: (len(footprints[i] & batch_fp), order_sizes[i]),
            )
            add(best)
            pool.remove(best)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_footprints(
        orders: list[dict[int, int]], aisles: list[dict[int, int]]
    ) -> list[set[int]]:
        item_to_aisles: dict[int, set[int]] = {}
        for a_idx, aisle in enumerate(aisles):
            for it in aisle:
                item_to_aisles.setdefault(it, set()).add(a_idx)
        empty: set[int] = set()
        footprints: list[set[int]] = []
        for order in orders:
            fp: set[int] = set()
            for it in order:
                fp |= item_to_aisles.get(it, empty)
            footprints.append(fp)
        return footprints

    @staticmethod
    def _theta_values(footprints: list[set[int]], n_aisles: int) -> list[float]:
        denom = max(1, n_aisles)
        theta: list[float] = []
        for fp in footprints:
            if not fp:
                theta.append(0.0)
            else:
                theta.append(sum(fp) / len(fp) / denom)
        return theta

    @staticmethod
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
