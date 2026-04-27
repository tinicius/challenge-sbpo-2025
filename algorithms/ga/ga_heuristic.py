import time
import numpy as np
from mealpy import IntegerVar, Termination
from mealpy.evolutionary_based import GA

from algorithms.base import Algorithm
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from problems.base import ProblemInput
from algorithms.utils.similarity import similarity
from algorithms.utils.local_search_aisle import (
    VALID_OPERATORS as _LS_OPERATORS,
    VALID_STRATEGIES as _LS_STRATEGIES,
    local_search_pass,
)

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
_VALID_LS_MODES = {"off", "on_seeds", "final", "both", "periodic"}

_EMPTY_RESULT = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}
_LB_PENALTY_WEIGHT = 0.1
_MAX_SEED_TIME_SECONDS = 3.0
_DEFAULT_SEED_TIME_SECONDS = 1.5
_MAX_SEED_ANCHORS = 12

_LS_DEFAULTS = {
    "mode": "off",
    "operators": ["remove", "swap", "add"],
    "strategy": "first_improvement",
    "max_iterations": 200,
    "elite_size": 1,
    "period": 25,
    "time_fraction": 0.2,
    "neighbor_cap": 50,
}


class GeneticAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "ga_aisle_based"

    def __init__(self, params: dict):
        super().__init__(params)

        variant = params.get("variant", "BaseGA")
        if variant not in _VARIANT_MAP:
            raise ValueError(f"GeneticAlgorithm: invalid 'variant'={variant!r}")

        self._variant = variant
        self._pop_size = int(params.get("pop_size", 50))
        self._epoch = int(params.get("epoch", 200))
        self._pc = float(params.get("pc", 0.9))
        self._pm = float(params.get("pm", 0.05))
        self._selection = params.get("selection", "tournament")
        self._crossover = params.get("crossover", "uniform")
        self._mutation = params.get("mutation", "flip")
        self._k_way = float(params.get("k_way", 0.2))
        self._seed_with_heuristics = bool(params.get("seed_with_heuristics", True))
        self._start = params.get("start")

        time_budget = params.get("time_budget")
        self._time_budget = float(time_budget) if time_budget is not None else None

        self._seed = params.get("seed")
        self._elite_best = float(params.get("elite_best", 0.1))
        self._elite_worst = float(params.get("elite_worst", 0.3))

        ls_cfg = dict(_LS_DEFAULTS)
        ls_cfg.update(params.get("local_search") or {})
        if ls_cfg["mode"] not in _VALID_LS_MODES:
            raise ValueError(
                f"GeneticAlgorithm: invalid local_search.mode={ls_cfg['mode']!r}; "
                f"expected one of {sorted(_VALID_LS_MODES)}"
            )
        if ls_cfg["strategy"] not in _LS_STRATEGIES:
            raise ValueError(
                f"GeneticAlgorithm: invalid local_search.strategy={ls_cfg['strategy']!r}"
            )
        bad_ops = [o for o in ls_cfg["operators"] if o not in _LS_OPERATORS]
        if bad_ops:
            raise ValueError(f"GeneticAlgorithm: invalid local_search operators {bad_ops}")
        self._ls_mode = ls_cfg["mode"]
        self._ls_operators = list(ls_cfg["operators"])
        self._ls_strategy = ls_cfg["strategy"]
        self._ls_max_iter = int(ls_cfg["max_iterations"])
        self._ls_elite_size = max(1, int(ls_cfg["elite_size"]))
        self._ls_period = max(1, int(ls_cfg["period"]))
        self._ls_time_fraction = float(ls_cfg["time_fraction"])
        self._ls_neighbor_cap = int(ls_cfg["neighbor_cap"])

        self.last_best = dict(_EMPTY_RESULT)

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)

        n_aisles = instance.nAisles
        if n_aisles == 0 or instance.nOrders == 0:
            return dict(_EMPTY_RESULT)

        solve_start = time.time()

        # Precompute Data Structures for maximum speed
        order_sizes = [sum(o.values()) for o in instance.orders]
        unique_items = list(set(item for a in instance.aisles for item in a.keys()))
        item_to_col = {item: idx for idx, item in enumerate(unique_items)}
        num_items = len(unique_items)

        # C-Contiguous arrays for faster memory access
        aisle_matrix = np.zeros((n_aisles, num_items), dtype=int)
        for i, aisle in enumerate(instance.aisles):
            for item, qty in aisle.items():
                aisle_matrix[i, item_to_col[item]] = qty

        order_matrix = np.zeros((instance.nOrders, num_items), dtype=int)
        for idx, order in enumerate(instance.orders):
            for item, qty in order.items():
                order_matrix[idx, item_to_col[item]] = qty

        self.order_priorities = sorted(
            range(instance.nOrders), key=lambda idx: order_sizes[idx], reverse=True
        )

        # FAST CACHE: Pre-extract ordered sizes and matrices to standard Python lists
        # of NumPy arrays to avoid slicing overhead inside the fitness loop
        ordered_sizes_fast = [order_sizes[i] for i in self.order_priorities]
        ordered_matrix_fast = [order_matrix[i] for i in self.order_priorities]

        fitness = self._make_fitness(
            instance, n_aisles, aisle_matrix, ordered_sizes_fast, ordered_matrix_fast
        )
        fitness_cache = fitness.cache

        # LS helper: derive selected_orders from a mask. Reuses the cache
        # populated by `fitness` (a mask reaching LS was just evaluated by GA
        # or fitness_fn), avoiding a duplicate pack-orders pass.
        def _orders_for_mask(mask: np.ndarray) -> list[int]:
            active = np.flatnonzero(mask > 0.5)
            if active.size == 0:
                return []
            t = tuple(active)
            cached = fitness_cache.get(t)
            if cached is not None:
                return cached[1]
            # Cold-cache fallback: trigger fitness to populate the entry.
            fitness(mask)
            cached = fitness_cache.get(t)
            return cached[1] if cached else []

        # Precompute once per solve() — used by LS to skip O(n_orders) work
        # in unmet-demand subtraction and O(items) work per similarity call.
        total_demand: dict[int, int] = {}
        for order in instance.orders:
            for item, qty in order.items():
                total_demand[item] = total_demand.get(item, 0) + qty
        aisle_key_sets = [frozenset(a.keys()) for a in instance.aisles]

        # Time budget split: reserve `ls_total` for LS, give the rest to mealpy.
        ls_total, ga_budget = self._compute_time_split(solve_start)

        # Build seeds; optionally polish them with LS.
        try:
            starting = self._build_starting_solutions(instance, n_aisles)
        except Exception as exc:
            print(f"GA: error building starting solutions: {exc}")
            starting = None

        on_seeds_budget = 0.0
        final_budget = 0.0
        if self._ls_mode == "on_seeds":
            on_seeds_budget = ls_total
        elif self._ls_mode == "final":
            final_budget = ls_total
        elif self._ls_mode == "both":
            on_seeds_budget = ls_total * 0.5
            final_budget = ls_total * 0.5

        if (
            self._ls_mode in {"on_seeds", "both"}
            and starting
            and self._ls_max_iter > 0
        ):
            self._polish_seeds(
                starting,
                fitness,
                instance,
                _orders_for_mask,
                on_seeds_budget,
                total_demand,
                aisle_key_sets,
            )

        # Run GA: single shot or chunked (periodic LS).
        try:
            if self._ls_mode == "periodic":
                self._run_periodic(
                    starting,
                    fitness,
                    instance,
                    n_aisles,
                    _orders_for_mask,
                    ls_total,
                    ga_budget,
                    solve_start,
                    total_demand,
                    aisle_key_sets,
                )
            else:
                self._run_single(starting, fitness, n_aisles, ga_budget)
        except Exception as exc:
            print(f"GA encountered an error: {exc}")

        # Final polish on the incumbent.
        if self._ls_mode in {"final", "both"} and self._ls_max_iter > 0:
            self._polish_final(
                fitness,
                instance,
                n_aisles,
                _orders_for_mask,
                final_budget,
                total_demand,
                aisle_key_sets,
            )

        if self.last_best["selected_orders"]:
            return dict(self.last_best)

        return dict(_EMPTY_RESULT)

    # ---------- Time-budget helpers ----------------------------------------

    def _compute_time_split(self, start: float) -> tuple[float, float | None]:
        """Return `(ls_total, ga_budget)` in seconds."""
        if self._time_budget is None:
            return 0.0, None
        if self._ls_mode == "off":
            return 0.0, self._time_budget
        ls_total = max(0.0, min(0.95, self._ls_time_fraction)) * self._time_budget
        ga_budget = max(0.0, self._time_budget - ls_total)
        return ls_total, ga_budget

    def _remaining_deadline(self, budget: float) -> float | None:
        if budget <= 0:
            return None
        return time.time() + budget

    # ---------- GA driver ---------------------------------------------------

    def _build_model_kwargs(self, epoch: int) -> dict:
        kwargs = dict(
            epoch=epoch,
            pop_size=self._pop_size,
            pc=self._pc,
            pm=self._pm,
            selection=self._selection,
            crossover=self._crossover,
            mutation=self._mutation,
            k_way=self._k_way,
        )
        if self._variant.startswith("Elite"):
            kwargs["elite_best"] = self._elite_best
            kwargs["elite_worst"] = self._elite_worst
        return kwargs

    def _build_problem_def(self, fitness, n_aisles: int) -> dict:
        return {
            "obj_func": fitness,
            "bounds": [IntegerVar(lb=0, ub=1) for _ in range(n_aisles)],
            "minmax": "max",
            "log_to": None,
        }

    def _run_single(
        self,
        starting,
        fitness,
        n_aisles: int,
        ga_budget: float | None,
    ) -> None:
        model = _VARIANT_MAP[self._variant](**self._build_model_kwargs(self._epoch))
        problem_def = self._build_problem_def(fitness, n_aisles)
        solve_kwargs = {}
        if starting:
            solve_kwargs["starting_solutions"] = np.asarray(starting, dtype=int)
        if self._seed is not None:
            solve_kwargs["seed"] = int(self._seed)
        if ga_budget is not None and ga_budget > 0:
            solve_kwargs["termination"] = Termination(max_time=ga_budget)
        model.solve(problem_def, **solve_kwargs)

    def _run_periodic(
        self,
        starting,
        fitness,
        instance: ProblemInput,
        n_aisles: int,
        orders_for_mask,
        ls_total: float,
        ga_budget: float | None,
        solve_start: float,
        total_demand: dict[int, int],
        aisle_key_sets: list[frozenset],
    ) -> None:
        chunk = self._ls_period
        total_epochs = self._epoch
        n_chunks = max(1, (total_epochs + chunk - 1) // chunk)
        per_chunk_ls = (ls_total / n_chunks) if ls_total > 0 else 0.0

        seeds = list(starting) if starting else []
        done = 0
        chunk_idx = 0
        while done < total_epochs:
            this_chunk = min(chunk, total_epochs - done)
            model = _VARIANT_MAP[self._variant](
                **self._build_model_kwargs(this_chunk)
            )
            problem_def = self._build_problem_def(fitness, n_aisles)
            solve_kwargs = {}
            if seeds:
                solve_kwargs["starting_solutions"] = np.asarray(seeds, dtype=int)
            if self._seed is not None:
                solve_kwargs["seed"] = int(self._seed) + chunk_idx
            if ga_budget is not None:
                remaining_ga = ga_budget - (time.time() - solve_start)
                if remaining_ga <= 0:
                    break
                solve_kwargs["termination"] = Termination(max_time=remaining_ga)

            model.solve(problem_def, **solve_kwargs)
            done += this_chunk
            chunk_idx += 1

            # Build next-chunk seed pool: top elite_size individuals polished by LS,
            # then fill with the rest of the population.
            try:
                pop = self._extract_population_masks(model, n_aisles)
            except Exception:
                pop = []

            polished: list[np.ndarray] = []
            elite = pop[: self._ls_elite_size]
            for mask in elite:
                if per_chunk_ls > 0 and self._ls_max_iter > 0:
                    deadline = time.time() + per_chunk_ls / max(1, len(elite))
                    new_mask, _, _ = local_search_pass(
                        mask=mask,
                        fitness_fn=fitness,
                        aisles=instance.aisles,
                        orders=instance.orders,
                        operators=self._ls_operators,
                        strategy=self._ls_strategy,
                        max_iterations=self._ls_max_iter,
                        neighbor_cap=self._ls_neighbor_cap,
                        deadline=deadline,
                        selected_orders_fn=orders_for_mask,
                        total_demand=total_demand,
                        aisle_key_sets=aisle_key_sets,
                    )
                    polished.append(new_mask)
                else:
                    polished.append(mask)

            seeds = polished + pop[self._ls_elite_size :]
            if not seeds:
                break

    def _extract_population_masks(self, model, n_aisles: int) -> list[np.ndarray]:
        pop = getattr(model, "pop", None) or []
        sorted_pop = sorted(
            pop,
            key=lambda a: getattr(getattr(a, "target", None), "fitness", 0.0),
            reverse=True,
        )
        out: list[np.ndarray] = []
        for agent in sorted_pop:
            sol = np.asarray(getattr(agent, "solution", []), dtype=float)
            if sol.size != n_aisles:
                continue
            out.append((sol > 0.5).astype(int))
        return out

    # ---------- Local-search wrappers --------------------------------------

    def _polish_seeds(
        self,
        seeds: list[np.ndarray],
        fitness,
        instance: ProblemInput,
        orders_for_mask,
        budget: float,
        total_demand: dict[int, int],
        aisle_key_sets: list[frozenset],
    ) -> None:
        if not seeds or self._ls_max_iter <= 0:
            return
        per_seed = (budget / len(seeds)) if budget > 0 else 0.0
        for i, seed in enumerate(seeds):
            mask = np.asarray(seed, dtype=int)
            if mask.ndim != 1:
                continue
            deadline = time.time() + per_seed if per_seed > 0 else None
            new_mask, _, _ = local_search_pass(
                mask=mask,
                fitness_fn=fitness,
                aisles=instance.aisles,
                orders=instance.orders,
                operators=self._ls_operators,
                strategy=self._ls_strategy,
                max_iterations=self._ls_max_iter,
                neighbor_cap=self._ls_neighbor_cap,
                deadline=deadline,
                selected_orders_fn=orders_for_mask,
                total_demand=total_demand,
                aisle_key_sets=aisle_key_sets,
            )
            seeds[i] = new_mask

    def _polish_final(
        self,
        fitness,
        instance: ProblemInput,
        n_aisles: int,
        orders_for_mask,
        budget: float,
        total_demand: dict[int, int],
        aisle_key_sets: list[frozenset],
    ) -> None:
        visited = self.last_best.get("visited_aisles") or []
        if not visited:
            return
        mask = np.zeros(n_aisles, dtype=int)
        mask[np.asarray(visited, dtype=int)] = 1
        deadline = time.time() + budget if budget > 0 else None
        local_search_pass(
            mask=mask,
            fitness_fn=fitness,
            aisles=instance.aisles,
            orders=instance.orders,
            operators=self._ls_operators,
            strategy=self._ls_strategy,
            max_iterations=self._ls_max_iter,
            neighbor_cap=self._ls_neighbor_cap,
            deadline=deadline,
            selected_orders_fn=orders_for_mask,
            total_demand=total_demand,
            aisle_key_sets=aisle_key_sets,
        )

    # ---------- Fitness ----------------------------------------------------

    def _make_fitness(
        self, instance, n_aisles, aisle_matrix, ordered_sizes_fast, ordered_matrix_fast
    ):
        lb, ub = instance.lb, instance.ub
        cache = {}

        # Avoid `self.` lookups inside loop
        order_priorities = self.order_priorities

        def fitness(x):
            # Extremely fast binary threshold and index extraction
            active_indices = np.nonzero(x > 0.5)[0]

            if len(active_indices) == 0:
                return 0.0

            # Tiny tuple hashes instantly compared to full binary array
            t_x = tuple(active_indices)
            cached = cache.get(t_x)
            if cached is not None:
                return cached[0]

            # Fast matrix sum (skips multiplying zeros entirely)
            current_stock = np.sum(aisle_matrix[active_indices], axis=0)

            selected_orders = []
            total_volume = 0

            # C-speed iteration using zip. No array slicing happens here!
            for idx, size, order_req in zip(
                order_priorities, ordered_sizes_fast, ordered_matrix_fast
            ):
                if total_volume + size > ub:
                    continue

                # Fast numpy comparison
                if np.all(order_req <= current_stock):
                    selected_orders.append(idx)
                    total_volume += size
                    current_stock -= order_req

            n_used = len(active_indices)

            # Penalize sub-LB solutions heavily and return early
            if total_volume < lb:
                penalty = (total_volume / lb) ** 2
                obj = (total_volume / n_used) * penalty * _LB_PENALTY_WEIGHT
                cache[t_x] = (obj, selected_orders, [])
                return obj

            # Only prune IF solution meets LB (Pruning is expensive)
            real_demand: dict[int, int] = {}
            for idx in selected_orders:
                for item, q in instance.orders[idx].items():
                    real_demand[item] = real_demand.get(item, 0) + q

            used_aisles = _greedy_cover(
                real_demand, list(active_indices), instance.aisles
            )
            n_used_pruned = len(used_aisles) if used_aisles else n_used

            obj = total_volume / n_used_pruned

            # Track best solution manually
            if obj > self.last_best["objective"]:
                self.last_best = {
                    "selected_orders": sorted(selected_orders),
                    "visited_aisles": sorted(used_aisles),
                    "objective": obj,
                }

            cache[t_x] = (obj, selected_orders, used_aisles)
            return obj

        # Expose the cache so callers (e.g. LS) can reuse the order packing
        # already computed by `fitness` instead of recomputing it.
        fitness.cache = cache
        return fitness

    # ---------- Starting population ---------------------------------------

    def _build_starting_solutions(self, instance: ProblemInput, n_aisles: int):
        if not self._seed_with_heuristics:
            return None

        heuristics_seeds = [
            SimpleHeuristic(
                {
                    "order": "desc",
                    "greedy": "multi",
                    "first_order": "bigger",
                    "exact": True,
                }
            ).solve(instance),
            AisleFirstHeuristic(
                {"score": "useful", "order": "desc", "prune": "multi"}
            ).solve(instance),
        ]

        # Keep seeding cheap so GA iterations consume most of the time budget.
        if self._time_budget is not None:
            seed_time_limit = min(self._time_budget * 0.10, _MAX_SEED_TIME_SECONDS)
        else:
            seed_time_limit = _DEFAULT_SEED_TIME_SECONDS

        start_time = time.time()
        seeds = []

        # Track the best heuristic result so the GA never returns worse than it.
        for result in heuristics_seeds:
            visited = result.get("visited_aisles") or []
            if visited:
                seeds.append(np.array(visited))
            if result.get("objective", 0.0) > self.last_best["objective"]:
                self.last_best = {
                    "selected_orders": sorted(result.get("selected_orders", [])),
                    "visited_aisles": sorted(visited),
                    "objective": float(result["objective"]),
                }

        if self._start == "random":
            seeds.extend(
                self.get_random_seeds(
                    instance, n_aisles, self._pop_size, start_time, seed_time_limit
                )
            )
        elif self._start == "seed_aisle":
            seeds.extend(
                self.get_seed_aisles_seeds(
                    instance,
                    n_aisles,
                    start_time=start_time,
                    time_limit=seed_time_limit,
                )
            )
        else:
            seeds.extend(
                self._mixed_seeds(instance, n_aisles, start_time, seed_time_limit)
            )

        # Pad remaining population with random variations if needed
        if 0 < len(seeds) < self._pop_size:
            random_seeds = self.get_random_seeds(
                instance,
                n_aisles,
                self._pop_size - len(seeds),
                start_time,
                seed_time_limit,
            )
            seeds.extend(random_seeds)

        masks = []
        for s in seeds[: self._pop_size]:
            chrom = np.zeros(n_aisles, dtype=int)
            chrom[s] = 1
            masks.append(chrom)

        if not masks:
            return None

        # mealpy requires len(starting_solutions) == pop_size.
        if len(masks) < self._pop_size:
            rng = np.random.default_rng(self._seed)
            for _ in range(self._pop_size - len(masks)):
                masks.append(rng.integers(0, 2, size=n_aisles, dtype=int))

        if len(masks) > self._pop_size:
            masks = masks[: self._pop_size]

        return masks

    def _mixed_seeds(
        self,
        instance: ProblemInput,
        n_aisles: int,
        start_time: float,
        time_limit: float,
    ) -> list[np.ndarray]:
        seeds: list[np.ndarray] = []

        greedy = self._greedy_demand_seed(instance, n_aisles)
        if greedy is not None:
            seeds.append(greedy)

        if len(seeds) < self._pop_size:
            seeds.extend(
                self.get_seed_aisles_seeds(
                    instance, n_aisles, start_time=start_time, time_limit=time_limit
                )
            )

        if len(seeds) < self._pop_size:
            seeds.extend(
                self.get_random_seeds(
                    instance,
                    n_aisles,
                    self._pop_size - len(seeds),
                    start_time,
                    time_limit,
                )
            )

        return seeds

    def _greedy_demand_seed(
        self, instance: ProblemInput, n_aisles: int
    ) -> np.ndarray | None:
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
                return np.array(selected_aisles)

        return None

    def get_seed_aisles_seeds(
        self,
        instance: ProblemInput,
        n_aisles: int,
        start_time: float | None = None,
        time_limit: float | None = None,
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        ordered_orders = _orders_by_size_desc(instance)

        # Limit expensive anchor-based seeding. Extra anchors bring little diversity
        # relative to the runtime overhead of repeated similarity sorts.
        anchor_count = min(self._pop_size, n_aisles, _MAX_SEED_ANCHORS)
        if anchor_count <= 0:
            return out

        anchor_candidates = sorted(
            range(n_aisles),
            key=lambda idx: sum(instance.aisles[idx].values()),
            reverse=True,
        )[:anchor_count]

        for anchor in anchor_candidates:
            if (
                start_time is not None
                and time_limit is not None
                and (time.time() - start_time) > time_limit
            ):
                break

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
                out.append(np.array(selected_aisles))

        return out

    def get_random_seeds(
        self,
        instance: ProblemInput,
        n_aisles: int,
        n_seeds: int,
        start_time: float,
        time_limit: float,
    ) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        base_seed = self._seed or 0

        for i in range(n_seeds):
            # Anti-timeout mechanism: abort seeding if it takes too long
            if time.time() - start_time > time_limit:
                break
            try:
                result = SimpleHeuristic(
                    {"greedy": "simple", "seed": base_seed + i}
                ).solve(instance)
                visited = result.get("visited_aisles") or []
                if visited:
                    out.append(np.array(visited))
            except Exception:
                continue

        return out


# ---------- Module-level helpers ------------------------------------------


def _orders_by_size_desc(instance: ProblemInput) -> list[int]:
    return sorted(
        range(instance.nOrders),
        key=lambda idx: sum(instance.orders[idx].values()),
        reverse=True,
    )


def _greedy_fill_orders(
    ordered_orders: list[int], instance: ProblemInput, stock: dict[int, int], ub: int
) -> tuple[list[int], int]:
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
    demand: dict[int, int], candidate_aisles: list[int], aisles: list[dict[int, int]]
) -> list[int]:
    remaining = {it: q for it, q in demand.items() if q > 0}
    available = list(candidate_aisles)
    used: list[int] = []

    while remaining and available:
        best_pos, best_score = -1, 0
        for pos, a_idx in enumerate(available):
            score = sum(min(remaining.get(it, 0), q) for it, q in aisles[a_idx].items())
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
