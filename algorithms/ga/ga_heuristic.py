import numpy as np
from mealpy import BinaryVar, Termination
from mealpy.evolutionary_based import GA

from algorithms.base import Algorithm
from algorithms.simple.simple_heuristic import SimpleHeuristic
from problems.base import ProblemInput

from algorithms.utils.similarity import similarity

_VARIANT_MAP = {
    "BaseGA": GA.BaseGA,
    "SingleGA": GA.SingleGA,
    "MultiGA": GA.MultiGA,
    "EliteSingleGA": GA.EliteSingleGA,
    "EliteMultiGA": GA.EliteMultiGA,
}

_VALID_SELECTION = {"tournament", "roulette", "random"}
_VALID_CROSSOVER = {"one_point", "multi_points", "uniform", "arithmetic"}
_VALID_MUTATION = {"flip", "swap", "scramble", "inversion"}
_VALID_START = {None, "random", "seed_aisle"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

# Sub-LB penalty multiplier — small enough that any feasible solution
# strictly dominates any infeasible one, large enough to give the GA
# a usable gradient toward LB.
_LB_PENALTY_WEIGHT = 0.1


class GeneticAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "ga_aisle_based"

    def __init__(self, params: dict):
        super().__init__(params)

        variant = params.get("variant", "BaseGA")
        if variant not in _VARIANT_MAP:
            raise ValueError(
                f"GeneticAlgorithm: invalid 'variant'={variant!r}; "
                f"expected one of {sorted(_VARIANT_MAP)}"
            )
        selection = params.get("selection", "tournament")
        if selection not in _VALID_SELECTION:
            raise ValueError(
                f"GeneticAlgorithm: invalid 'selection'={selection!r}; "
                f"expected one of {sorted(_VALID_SELECTION)}"
            )
        crossover = params.get("crossover", "uniform")
        if crossover not in _VALID_CROSSOVER:
            raise ValueError(
                f"GeneticAlgorithm: invalid 'crossover'={crossover!r}; "
                f"expected one of {sorted(_VALID_CROSSOVER)}"
            )
        mutation = params.get("mutation", "flip")
        if mutation not in _VALID_MUTATION:
            raise ValueError(
                f"GeneticAlgorithm: invalid 'mutation'={mutation!r}; "
                f"expected one of {sorted(_VALID_MUTATION)}"
            )
        start = params.get("start")
        if start not in _VALID_START:
            raise ValueError(
                f"GeneticAlgorithm: invalid 'start'={start!r}; "
                f"expected one of {sorted(s for s in _VALID_START if s)} or None"
            )

        self._variant = variant
        self._pop_size = int(params.get("pop_size", 50))
        self._epoch = int(params.get("epoch", 200))
        self._pc = float(params.get("pc", 0.9))
        self._pm = float(params.get("pm", 0.05))
        self._selection = selection
        self._crossover = crossover
        self._mutation = mutation
        self._k_way = float(params.get("k_way", 0.2))
        self._seed_with_heuristics = bool(params.get("seed_with_heuristics", True))
        self._start = start

        time_budget = params.get("time_budget")
        self._time_budget = float(time_budget) if time_budget is not None else None

        self._seed = params.get("seed")
        self._elite_best = float(params.get("elite_best", 0.1))
        self._elite_worst = float(params.get("elite_worst", 0.3))

        self.last_best = dict(_EMPTY_RESULT)

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)

        n_aisles = instance.nAisles
        if n_aisles == 0 or instance.nOrders == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in instance.orders]

        fitness = self._make_fitness(instance, order_sizes, n_aisles)

        problem_def = {
            "obj_func": fitness,
            "bounds": BinaryVar(n_vars=n_aisles),
            "minmax": "max",
            "log_to": None,
        }

        model_kwargs = dict(
            epoch=self._epoch,
            pop_size=self._pop_size,
            pc=self._pc,
            pm=self._pm,
            selection=self._selection,
            crossover=self._crossover,
            mutation=self._mutation,
            k_way=self._k_way,
        )

        if self._variant.startswith("Elite"):
            model_kwargs["elite_best"] = self._elite_best
            model_kwargs["elite_worst"] = self._elite_worst

        model = _VARIANT_MAP[self._variant](**model_kwargs)

        solve_kwargs = {}

        try:
            starting = self._build_starting_solutions(instance, n_aisles)
        except Exception as exc:
            print(f"GA: error building starting solutions: {exc}")
            starting = None

        if starting:
            solve_kwargs["starting_solutions"] = np.asarray(starting, dtype=int)
        if self._seed is not None:
            solve_kwargs["seed"] = int(self._seed)
        if self._time_budget is not None:
            solve_kwargs["termination"] = Termination(max_time=self._time_budget)

        try:
            model.solve(problem_def, **solve_kwargs)
        except Exception as exc:
            print(f"GA encountered an error: {exc}")

        if self.last_best["selected_orders"]:
            return dict(self.last_best)

        return dict(_EMPTY_RESULT)

    # ---------- Fitness ----------------------------------------------------

    def _make_fitness(self, instance, order_sizes, n_aisles):
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders

        def fitness(x):
            # 1. Decode chromosome.
            proposed_aisles = [i for i in range(n_aisles) if x[i] > 0.5]
            if not proposed_aisles:
                return 0.0

            # 2. Aggregate available stock from proposed aisles.
            available_stock: dict[int, int] = {}
            for i in proposed_aisles:
                for item, qty in aisles[i].items():
                    available_stock[item] = available_stock.get(item, 0) + qty

            # 3. Greedy fill: largest orders first, respecting UB and stock.
            order_priorities = sorted(
                range(n_orders), key=lambda idx: order_sizes[idx], reverse=True
            )

            selected_orders: list[int] = []
            total_volume = 0
            stock = dict(available_stock)

            for idx in order_priorities:
                size = order_sizes[idx]
                if size == 0 or total_volume + size > ub:
                    continue
                order = orders[idx]
                if any(stock.get(item, 0) < q for item, q in order.items()):
                    continue
                selected_orders.append(idx)
                total_volume += size
                for item, q in order.items():
                    stock[item] -= q

            if not selected_orders:
                return 0.0

            # 4. Prune unused aisles via greedy max-contribution cover so that
            #    a chromosome that proposes extra useless aisles is not
            #    penalized for them.
            real_demand: dict[int, int] = {}
            for idx in selected_orders:
                for item, q in orders[idx].items():
                    real_demand[item] = real_demand.get(item, 0) + q

            used_aisles = _greedy_cover(real_demand, proposed_aisles, aisles)
            n_used = len(used_aisles) if used_aisles else len(proposed_aisles)

            # 5. Sub-LB penalty: keep a smooth gradient toward LB but never
            #    let an infeasible solution beat a feasible one.
            if total_volume < lb:
                penalty = (total_volume / lb) ** 2
                return (total_volume / n_used) * penalty * _LB_PENALTY_WEIGHT

            obj = total_volume / n_used

            if obj > self.last_best["objective"]:
                self.last_best = {
                    "selected_orders": sorted(selected_orders),
                    "visited_aisles": sorted(used_aisles),
                    "objective": obj,
                }

            return obj

        return fitness

    # ---------- Starting population ---------------------------------------

    def _build_starting_solutions(self, instance: ProblemInput, n_aisles: int):
        if not self._seed_with_heuristics:
            return None

        if self._start == "random":
            seeds = list(self.get_random_seeds(instance, n_aisles, self._pop_size))
        elif self._start == "seed_aisle":
            seeds = list(self.get_seed_aisles_seeds(instance, n_aisles))
        else:
            seeds = self._mixed_seeds(instance, n_aisles)

        # De-duplicate while preserving order.
        unique: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for s in seeds:
            arr = np.asarray(s, dtype=int).ravel()
            if arr.shape[0] != n_aisles or arr.sum() == 0:
                continue
            key = tuple(int(v) for v in arr)
            if key in seen:
                continue
            seen.add(key)
            unique.append(arr)
            if len(unique) >= self._pop_size:
                break

        # mealpy requires len(starting_solutions) == pop_size. Pad with
        # jittered copies of the best known seed to keep diversity.
        if 0 < len(unique) < self._pop_size:
            rng = np.random.default_rng(self._seed)
            base = unique[0]
            while len(unique) < self._pop_size:
                jitter = rng.random(n_aisles) < 0.1
                candidate = np.where(jitter, 1 - base, base).astype(int)
                unique.append(candidate)

        return unique or None

    def _mixed_seeds(
        self, instance: ProblemInput, n_aisles: int
    ) -> list[np.ndarray]:
        """Default seeding: one greedy demand-aware seed + similarity seeds + random."""
        seeds: list[np.ndarray] = []

        greedy = self._greedy_demand_seed(instance, n_aisles)
        if greedy is not None:
            seeds.append(greedy)

        if len(seeds) < self._pop_size:
            seeds.extend(self.get_seed_aisles_seeds(instance, n_aisles))

        if len(seeds) < self._pop_size:
            seeds.extend(
                self.get_random_seeds(
                    instance, n_aisles, self._pop_size - len(seeds)
                )
            )

        return seeds

    def _greedy_demand_seed(
        self, instance: ProblemInput, n_aisles: int
    ) -> np.ndarray | None:
        """Grow an aisle set ranked by total useful supply until LB is met."""
        total_demand: dict[int, int] = {}
        for order in instance.orders:
            for item, qty in order.items():
                total_demand[item] = total_demand.get(item, 0) + qty

        ordered_orders = _orders_by_size_desc(instance)
        ordered_aisles = sorted(
            range(n_aisles),
            key=lambda idx: sum(
                min(qty, total_demand.get(item, 0))
                for item, qty in instance.aisles[idx].items()
            ),
            reverse=True,
        )

        selected_aisles: list[int] = []
        stock: dict[int, int] = {}

        for next_aisle in ordered_aisles:
            selected_aisles.append(next_aisle)
            for item, qty in instance.aisles[next_aisle].items():
                stock[item] = stock.get(item, 0) + qty

            _, total_volume = _greedy_fill_orders(
                ordered_orders, instance, stock, instance.ub
            )
            if total_volume >= instance.lb:
                mask = np.zeros(n_aisles, dtype=int)
                for a in selected_aisles:
                    mask[a] = 1
                return mask

        return None

    def get_seed_aisles_seeds(
        self, instance: ProblemInput, n_aisles: int
    ) -> list[np.ndarray]:
        """One seed per anchor aisle, growing the set by similarity until LB."""
        out: list[np.ndarray] = []
        ordered_orders = _orders_by_size_desc(instance)

        anchor_count = min(self._pop_size, n_aisles)
        for anchor in range(anchor_count):
            ordered_aisles = sorted(
                range(n_aisles),
                key=lambda idx: similarity(
                    instance.aisles[anchor], instance.aisles[idx]
                ),
                reverse=True,
            )

            selected_aisles: list[int] = []
            stock: dict[int, int] = {}
            total_volume = 0

            for next_aisle in ordered_aisles:
                selected_aisles.append(next_aisle)
                for item, qty in instance.aisles[next_aisle].items():
                    stock[item] = stock.get(item, 0) + qty

                _, total_volume = _greedy_fill_orders(
                    ordered_orders, instance, stock, instance.ub
                )
                if total_volume >= instance.lb:
                    break

            if total_volume >= instance.lb:
                mask = np.zeros(n_aisles, dtype=int)
                for a in selected_aisles:
                    mask[a] = 1
                out.append(mask)

        return out

    def get_random_seeds(
        self, instance: ProblemInput, n_aisles: int, n_seeds: int = 50
    ) -> list[np.ndarray]:
        """Stochastic seeds via SimpleHeuristic with varying RNG seeds."""
        out: list[np.ndarray] = []
        base_seed = self._seed or 0
        for i in range(n_seeds):
            try:
                result = SimpleHeuristic(
                    {"greedy": "simple", "seed": base_seed + i}
                ).solve(instance)
            except Exception:
                continue
            visited = result.get("visited_aisles") or []
            if not visited:
                continue
            mask = np.zeros(n_aisles, dtype=int)
            for a in visited:
                if 0 <= a < n_aisles:
                    mask[a] = 1
            if mask.sum() > 0:
                out.append(mask)
        return out


# ---------- Module-level helpers ------------------------------------------


def _orders_by_size_desc(instance: ProblemInput) -> list[int]:
    return sorted(
        range(instance.nOrders),
        key=lambda idx: sum(instance.orders[idx].values()),
        reverse=True,
    )


def _greedy_fill_orders(
    ordered_orders: list[int],
    instance: ProblemInput,
    stock: dict[int, int],
    ub: int,
) -> tuple[list[int], int]:
    """Greedy fill respecting UB and stock. Mutates a local copy of stock."""
    selected: list[int] = []
    total_volume = 0
    local_stock = dict(stock)
    for o_idx in ordered_orders:
        order = instance.orders[o_idx]
        size = sum(order.values())
        if size == 0 or total_volume + size > ub:
            continue
        if any(local_stock.get(item, 0) < q for item, q in order.items()):
            continue
        selected.append(o_idx)
        total_volume += size
        for item, q in order.items():
            local_stock[item] -= q
    return selected, total_volume


def _greedy_cover(
    demand: dict[int, int],
    candidate_aisles: list[int],
    aisles: list[dict[int, int]],
) -> list[int]:
    """Pick the smallest subset of candidate_aisles that covers `demand`,
    greedily by max useful contribution."""
    remaining = {it: q for it, q in demand.items() if q > 0}
    available = list(candidate_aisles)
    used: list[int] = []

    while remaining and available:
        best_pos = -1
        best_score = 0
        for pos, a_idx in enumerate(available):
            score = sum(
                min(remaining.get(it, 0), q) for it, q in aisles[a_idx].items()
            )
            if score > best_score:
                best_score = score
                best_pos = pos

        if best_score == 0:
            break

        a_idx = available.pop(best_pos)
        used.append(a_idx)
        for it, q in aisles[a_idx].items():
            if it in remaining:
                remaining[it] -= q
                if remaining[it] <= 0:
                    del remaining[it]

    return used
