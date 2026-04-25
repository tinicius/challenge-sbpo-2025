import numpy as np
from mealpy import BinaryVar, Termination
from mealpy.evolutionary_based import GA

from algorithms.base import Algorithm
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.seed.seed_heuristic import SeedHeuristic
from problems.base import ProblemInput

from algorithms.utils.similarity import similarity
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select


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

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

# Sub-LB penalty multiplier — small enough that any feasible solution
# strictly dominates any infeasible one, large enough to give the GA
# a usable gradient toward LB.
_LB_PENALTY_WEIGHT = 0.1


class GeneticAlgorithmFull(Algorithm):
    @property
    def name(self) -> str:
        return "ga_full"

    def __init__(self, params: dict):
        super().__init__(params)

        variant = params.get("variant", "BaseGA")
        if variant not in _VARIANT_MAP:
            raise ValueError(
                f"GeneticAlgorithmFull: invalid 'variant'={variant!r}; "
                f"expected one of {sorted(_VARIANT_MAP)}"
            )
        selection = params.get("selection", "tournament")
        if selection not in _VALID_SELECTION:
            raise ValueError(
                f"GeneticAlgorithmFull: invalid 'selection'={selection!r}"
            )
        crossover = params.get("crossover", "uniform")
        if crossover not in _VALID_CROSSOVER:
            raise ValueError(
                f"GeneticAlgorithmFull: invalid 'crossover'={crossover!r}"
            )
        mutation = params.get("mutation", "flip")
        if mutation not in _VALID_MUTATION:
            raise ValueError(
                f"GeneticAlgorithmFull: invalid 'mutation'={mutation!r}"
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

        self.start = params.get("start", None)

        time_budget = params.get("time_budget")
        self._time_budget = float(time_budget) if time_budget else None

        self._seed = params.get("seed")
        self._elite_best = float(params.get("elite_best", 0.1))
        self._elite_worst = float(params.get("elite_worst", 0.3))

        self.last_best = dict(_EMPTY_RESULT)

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)

        n_aisles = instance.nAisles
        n_orders = instance.nOrders

        if n_aisles == 0 or n_orders == 0:
            return dict(_EMPTY_RESULT)

        order_sizes = [sum(o.values()) for o in instance.orders]

        fitness = self._make_fitness(instance, order_sizes, n_aisles, n_orders)

        problem_def = {
            "obj_func": fitness,
            "bounds": BinaryVar(n_vars=n_aisles + n_orders),
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
            starting = self._build_starting_solutions(instance)
        except Exception as exc:
            print(f"GA full: error building starting solutions: {exc}")
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
            print(f"GA full encountered an error: {exc}")
            return dict(self.last_best) if self.last_best["selected_orders"] else dict(_EMPTY_RESULT)

        if self.last_best["selected_orders"]:
            return dict(self.last_best)

        return dict(_EMPTY_RESULT)

    # ---------- Fitness ----------------------------------------------------

    def _make_fitness(self, instance, order_sizes, n_aisles, n_orders):
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub

        def fitness(x):
            # 1. Decode chromosome
            proposed_aisles = [i for i in range(n_aisles) if x[i] > 0.5]
            if not proposed_aisles:
                return 0.0

            proposed_orders = [
                i - n_aisles
                for i in range(n_aisles, n_aisles + n_orders)
                if x[i] > 0.5
            ]
            if not proposed_orders:
                return 0.0

            # 2. Aggregate stock from proposed aisles
            stock = {}
            for aisle_idx in proposed_aisles:
                for item, qty in aisles[aisle_idx].items():
                    stock[item] = stock.get(item, 0) + qty

            # 3. REPAIR: greedily filter the proposed orders so the result is
            #    always stock-feasible and within UB. Largest orders first
            #    favor higher numerator in the obj ratio.
            selected_orders = []
            total_volume = 0
            ordered = sorted(
                proposed_orders, key=lambda i: order_sizes[i], reverse=True
            )

            for idx in ordered:
                size = order_sizes[idx]
                if size == 0 or total_volume + size > ub:
                    continue
                order = orders[idx]
                if any(stock.get(it, 0) < q for it, q in order.items()):
                    continue
                selected_orders.append(idx)
                total_volume += size
                for it, q in order.items():
                    stock[it] -= q

            if not selected_orders:
                # Soft signal: at least the chromosome proposed valid aisles.
                return 1e-6

            # 4. Compute the actual aisles needed for the picked demand,
            #    restricted to the proposed aisle set (this is the GA's
            #    decision variable, after all).
            real_demand = {}
            for idx in selected_orders:
                for it, q in orders[idx].items():
                    real_demand[it] = real_demand.get(it, 0) + q

            used_aisles = self._used_aisles(real_demand, proposed_aisles, aisles)
            n_used = len(used_aisles) if used_aisles else len(proposed_aisles)

            # 5. Sub-LB penalty: keep gradient but never beat a feasible solution.
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

    @staticmethod
    def _used_aisles(
        demand: dict[int, int],
        proposed_aisles: list[int],
        aisles: list[dict[int, int]],
    ) -> list[int]:
        """Greedily pick the smallest subset of proposed_aisles that
        covers the demand. Largest contribution first."""
        remaining = {it: q for it, q in demand.items() if q > 0}
        available = list(proposed_aisles)
        used: list[int] = []

        while remaining and available:
            best_idx = -1
            best_score = 0
            for i, a_idx in enumerate(available):
                aisle = aisles[a_idx]
                score = sum(
                    min(remaining.get(it, 0), q) for it, q in aisle.items()
                )
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_score == 0:
                break

            a_idx = available.pop(best_idx)
            used.append(a_idx)
            for it, q in aisles[a_idx].items():
                if it in remaining:
                    remaining[it] -= q
                    if remaining[it] <= 0:
                        del remaining[it]

        return used

    # ---------- Starting population ---------------------------------------

    def _build_starting_solutions(self, instance: ProblemInput):
        if not self._seed_with_heuristics:
            return None

        n = instance.nAisles + instance.nOrders
        seeds: list[np.ndarray] = []

        # Always try the deterministic heuristic seeds first — they are the
        # most likely to be feasible right out of the gate.
        seeds.extend(self._heuristic_seeds(instance))

        if self.start == "random" or len(seeds) < self._pop_size:
            seeds.extend(self._random_heuristic_seeds(instance, self._pop_size))

        if self.start != "random":
            seeds.extend(self._aisle_similarity_seeds(instance))

        # De-duplicate and clip to pop size.
        unique: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for s in seeds:
            if s.shape[0] != n:
                continue
            key = tuple(int(v) for v in s)
            if key in seen:
                continue
            seen.add(key)
            unique.append(s)
            if len(unique) >= self._pop_size:
                break

        # mealpy requires len(starting_solutions) == pop_size, so pad with
        # randomized variants so the GA still gets diversity.
        if 0 < len(unique) < self._pop_size:
            rng = np.random.default_rng(self._seed)
            base = unique[0]
            while len(unique) < self._pop_size:
                # Flip ~10% of bits at random to seed diversity.
                jitter = rng.random(n) < 0.1
                candidate = np.where(jitter, 1 - base, base).astype(int)
                unique.append(candidate)

        return unique or None

    def _heuristic_seeds(self, instance: ProblemInput) -> list[np.ndarray]:
        """Diverse deterministic seeds from the available constructive heuristics."""
        configs = [
            (AisleFirstHeuristic, {"score": "useful", "order": "desc", "prune": "multi"}),
            (AisleFirstHeuristic, {"score": "useful", "order": "asc", "prune": "multi"}),
            (SeedHeuristic, {"seed_strategy": "biggest", "synergy": "min_new_aisles", "greedy": "multi"}),
            (SeedHeuristic, {"seed_strategy": "most_shared", "synergy": "min_new_aisles", "greedy": "multi"}),
            (SimpleHeuristic, {"order": "desc", "greedy": "multi"}),
        ]

        out: list[np.ndarray] = []
        for cls, params in configs:
            try:
                result = cls(params).solve(instance)
            except Exception:
                continue
            mask = self._mask_from_result(instance, result)
            if mask is not None:
                out.append(mask)
        return out

    def _random_heuristic_seeds(
        self, instance: ProblemInput, n_seeds: int
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(n_seeds):
            try:
                result = SimpleHeuristic(
                    {"greedy": "simple", "seed": (self._seed or 0) + i}
                ).solve(instance)
            except Exception:
                continue
            mask = self._mask_from_result(instance, result)
            if mask is not None:
                out.append(mask)
        return out

    def _aisle_similarity_seeds(self, instance: ProblemInput) -> list[np.ndarray]:
        """One seed per anchor aisle, growing the picked set by similarity."""
        out: list[np.ndarray] = []
        n_aisles = instance.nAisles

        ordered_orders = sorted(
            range(instance.nOrders),
            key=lambda idx: sum(instance.orders[idx].values()),
            reverse=True,
        )

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
            selected_orders: list[int] = []
            total_volume = 0

            for next_aisle in ordered_aisles:
                selected_aisles.append(next_aisle)

                stock: dict[int, int] = {}
                for a in selected_aisles:
                    for it, q in instance.aisles[a].items():
                        stock[it] = stock.get(it, 0) + q

                selected_orders = []
                total_volume = 0
                for o_idx in ordered_orders:
                    order = instance.orders[o_idx]
                    size = sum(order.values())
                    if total_volume + size > instance.ub:
                        continue
                    if any(stock.get(it, 0) < q for it, q in order.items()):
                        continue
                    selected_orders.append(o_idx)
                    total_volume += size
                    for it, q in order.items():
                        stock[it] -= q

                if total_volume >= instance.lb:
                    break

            if total_volume >= instance.lb and selected_orders:
                mask = self._mask_from_indices(
                    instance, selected_aisles, selected_orders
                )
                out.append(mask)

        return out

    @staticmethod
    def _mask_from_result(
        instance: ProblemInput, result: dict
    ) -> np.ndarray | None:
        visited = result.get("visited_aisles") or []
        selected = result.get("selected_orders") or []
        if not visited or not selected:
            return None
        return GeneticAlgorithmFull._mask_from_indices(instance, visited, selected)

    @staticmethod
    def _mask_from_indices(
        instance: ProblemInput,
        visited_aisles: list[int],
        selected_orders: list[int],
    ) -> np.ndarray:
        mask = np.zeros(instance.nAisles + instance.nOrders, dtype=int)
        for a in visited_aisles:
            if 0 <= a < instance.nAisles:
                mask[a] = 1
        for o in selected_orders:
            if 0 <= o < instance.nOrders:
                mask[instance.nAisles + o] = 1
        return mask
