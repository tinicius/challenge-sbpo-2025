import random

from algorithms.base import Algorithm
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class GeneticAlgorithmHeuristic(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "genetic_algorithm_heuristic"

    def _build_demand(
        self, chromosome: list[int]
    ) -> tuple[list[int], dict[int, int], int]:
        selected_orders: list[int] = []
        demand: dict[int, int] = {}
        units = 0

        for order_idx, bit in enumerate(chromosome):
            if bit == 0:
                continue

            selected_orders.append(order_idx)
            units += self.order_units[order_idx]

            for item, qty in self.orders[order_idx].items():
                demand[item] = demand.get(item, 0) + qty

        return selected_orders, demand, units

    def _compute_supply(self, aisle_indices: list[int]) -> dict[int, int]:
        supply: dict[int, int] = {}

        for aisle_idx in aisle_indices:
            for item, qty in self.aisles[aisle_idx].items():
                supply[item] = supply.get(item, 0) + qty

        return supply

    def _evaluate(self, chromosome: list[int]) -> dict:
        selected_orders, demand, units = self._build_demand(chromosome)

        if not selected_orders:
            return {
                "fitness": -self.penalty_scale,
                "objective": 0.0,
                "units": 0,
                "selected_orders": [],
                "visited_aisles": [],
                "feasible": False,
            }

        visited_aisles = multi_greedy_aisle_select(demand, self.aisles)

        supply = self._compute_supply(visited_aisles)
        stock_deficit = 0
        for item, req_qty in demand.items():
            stock_deficit += max(0, req_qty - supply.get(item, 0))

        lb_violation = max(0, self.lb - units)
        ub_violation = max(0, units - self.ub)

        raw_objective = 0.0
        if visited_aisles:
            raw_objective = units / len(visited_aisles)

        normalized_lb = lb_violation / max(1, self.lb)
        normalized_ub = ub_violation / max(1, self.ub)
        normalized_stock = stock_deficit / max(1, sum(demand.values()))

        penalty = (
            self.penalty_lb * normalized_lb
            + self.penalty_ub * normalized_ub
            + self.penalty_stock * normalized_stock
        )

        fitness = raw_objective - (self.penalty_scale * penalty)

        feasible = (
            self.lb <= units <= self.ub
            and stock_deficit == 0
            and len(visited_aisles) > 0
        )

        return {
            "fitness": fitness,
            "objective": raw_objective,
            "units": units,
            "selected_orders": selected_orders,
            "visited_aisles": visited_aisles,
            "feasible": feasible,
        }

    def _random_chromosome(self, rng: random.Random) -> list[int]:
        return [1 if rng.random() < 0.5 else 0 for _ in range(self.n_orders)]

    def _feasible_biased_chromosome(self, rng: random.Random) -> list[int]:
        chromosome = [0] * self.n_orders
        available = list(range(self.n_orders))
        rng.shuffle(available)

        total_units = 0

        for order_idx in available:
            order_units = self.order_units[order_idx]
            if total_units + order_units > self.ub:
                continue

            if rng.random() < 0.7:
                chromosome[order_idx] = 1
                total_units += order_units

        return chromosome

    def _repair_over_upper_bound(
        self, chromosome: list[int], units: int, rng: random.Random
    ) -> list[int]:
        if units <= self.ub:
            return chromosome

        active_orders = [idx for idx, bit in enumerate(chromosome) if bit == 1]
        active_orders.sort(key=lambda idx: (self.order_units[idx], rng.random()))

        repaired = list(chromosome)
        total_units = units

        for order_idx in active_orders:
            if total_units <= self.ub:
                break
            repaired[order_idx] = 0
            total_units -= self.order_units[order_idx]

        return repaired

    def _tournament_select(
        self, population: list[list[int]], evaluations: list[dict], rng: random.Random
    ) -> list[int]:
        sampled_indices = [
            rng.randrange(0, len(population)) for _ in range(self.tournament_size)
        ]

        best_idx = sampled_indices[0]
        for idx in sampled_indices[1:]:
            if evaluations[idx]["fitness"] > evaluations[best_idx]["fitness"]:
                best_idx = idx

        return list(population[best_idx])

    def _crossover(
        self, parent_a: list[int], parent_b: list[int], rng: random.Random
    ) -> tuple[list[int], list[int]]:
        if self.n_orders < 2 or rng.random() > self.crossover_rate:
            return list(parent_a), list(parent_b)

        child_a = []
        child_b = []

        for idx in range(self.n_orders):
            if rng.random() < 0.5:
                child_a.append(parent_a[idx])
                child_b.append(parent_b[idx])
            else:
                child_a.append(parent_b[idx])
                child_b.append(parent_a[idx])

        return child_a, child_b

    def _mutate(self, chromosome: list[int], rng: random.Random):
        for idx in range(self.n_orders):
            if rng.random() < self.mutation_rate:
                chromosome[idx] = 1 - chromosome[idx]

    def solve(self, instance: ProblemInput) -> dict:
        inst = self._prepare_instance(instance)
        self.n_orders = inst.nOrders
        self.n_aisles = inst.nAisles
        self.orders = inst.orders
        self.aisles = inst.aisles
        self.lb = inst.lb
        self.ub = inst.ub
        self.config = self.params

        # Config parsing (moved from old __init__)
        self.population_size = max(10, int(self.config.get("population_size", 80)))
        self.generations = max(1, int(self.config.get("generations", 120)))
        self.crossover_rate = float(self.config.get("crossover_rate", 0.85))
        self.tournament_size = max(2, int(self.config.get("tournament_size", 3)))
        self.elite_count = max(1, int(self.config.get("elite_count", 4)))
        self.penalty_scale = float(self.config.get("penalty_scale", 3.0))

        self.penalty_lb = float(self.config.get("penalty_lb", 1.0))
        self.penalty_ub = float(self.config.get("penalty_ub", 1.0))
        self.penalty_stock = float(self.config.get("penalty_stock", 2.0))

        configured_mutation = self.config.get("mutation_rate")
        if configured_mutation is None:
            dynamic = 1.0 / max(1, self.n_orders)
            self.mutation_rate = max(0.005, min(0.1, dynamic))
        else:
            self.mutation_rate = float(configured_mutation)

        self.repair_upper_bound = bool(self.config.get("repair_upper_bound", True))
        self.feasible_init_ratio = float(self.config.get("feasible_init_ratio", 0.6))

        self.order_units = [sum(order.values()) for order in self.orders]
        self.total_units_denominator = max(1, self.ub)

        if self.n_orders == 0 or self.n_aisles == 0:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        seed = self.config.get("seed")
        rng = random.Random(seed)

        population: list[list[int]] = []
        for _ in range(self.population_size):
            if rng.random() < self.feasible_init_ratio:
                chromosome = self._feasible_biased_chromosome(rng)
            else:
                chromosome = self._random_chromosome(rng)
            population.append(chromosome)

        cache: dict[tuple[int, ...], dict] = {}

        def evaluate_population(pop: list[list[int]]) -> list[dict]:
            evaluations: list[dict] = []
            for chromosome in pop:
                key = tuple(chromosome)
                eval_data = cache.get(key)
                if eval_data is None:
                    eval_data = self._evaluate(chromosome)
                    cache[key] = eval_data
                evaluations.append(eval_data)
            return evaluations

        evaluations = evaluate_population(population)

        best_feasible = None

        for eval_data in evaluations:
            if eval_data["feasible"]:
                if best_feasible is None:
                    best_feasible = eval_data
                elif eval_data["objective"] > best_feasible["objective"]:
                    best_feasible = eval_data

        for _ in range(self.generations):
            ranked_indices = sorted(
                range(len(population)),
                key=lambda idx: evaluations[idx]["fitness"],
                reverse=True,
            )

            next_population: list[list[int]] = []
            elite_limit = min(self.elite_count, len(population))
            for elite_idx in ranked_indices[:elite_limit]:
                next_population.append(list(population[elite_idx]))

            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population, evaluations, rng)
                parent_b = self._tournament_select(population, evaluations, rng)

                child_a, child_b = self._crossover(parent_a, parent_b, rng)

                self._mutate(child_a, rng)
                self._mutate(child_b, rng)

                if self.repair_upper_bound:
                    _, _, units_a = self._build_demand(child_a)
                    _, _, units_b = self._build_demand(child_b)
                    child_a = self._repair_over_upper_bound(child_a, units_a, rng)
                    child_b = self._repair_over_upper_bound(child_b, units_b, rng)

                next_population.append(child_a)
                if len(next_population) < self.population_size:
                    next_population.append(child_b)

            population = next_population
            evaluations = evaluate_population(population)

            for eval_data in evaluations:
                if not eval_data["feasible"]:
                    continue

                if best_feasible is None:
                    best_feasible = eval_data
                    continue

                if eval_data["objective"] > best_feasible["objective"]:
                    best_feasible = eval_data
                elif eval_data["objective"] == best_feasible["objective"] and len(
                    eval_data["visited_aisles"]
                ) < len(best_feasible["visited_aisles"]):
                    best_feasible = eval_data

        if best_feasible is None:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        selected_orders = sorted(set(best_feasible["selected_orders"]))
        visited_aisles = sorted(set(best_feasible["visited_aisles"]))

        if not selected_orders or not visited_aisles:
            return {'selected_orders': [], 'visited_aisles': [], 'objective': 0.0}

        total_items = sum(sum(inst.orders[o].values()) for o in selected_orders)
        objective = total_items / len(visited_aisles)
        return {'selected_orders': selected_orders, 'visited_aisles': visited_aisles, 'objective': objective}
