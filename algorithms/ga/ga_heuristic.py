import numpy as np
from mealpy import BinaryVar, Termination
from mealpy.evolutionary_based import GA

from algorithms.base import Algorithm
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from algorithms.seed.seed_heuristic import SeedHeuristic
from problems.base import ProblemInput

from algorithms.utils.similarity import similarity

# Usaremos a heurística para semear o GA inicial, mas agora mapeando para corredores
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


class GeneticAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "ga_aisle_based"

    def __init__(self, params: dict):
        super().__init__(params)

        # Validações de parâmetros mantidas idênticas ao original...
        self._variant = params.get("variant", "BaseGA")
        self._pop_size = int(params.get("pop_size", 50))
        self._epoch = int(params.get("epoch", 200))
        self._pc = float(params.get("pc", 0.9))
        self._pm = float(params.get("pm", 0.05))
        self._selection = params.get("selection", "tournament")
        self._crossover = params.get("crossover", "uniform")
        self._mutation = params.get("mutation", "flip")
        self._k_way = float(params.get("k_way", 0.2))
        self._seed_with_heuristics = params.get("seed_with_heuristics", True)

        self.start = params.get("start", None)

        time_budget = params.get("time_budget")
        self._time_budget = float(time_budget) if time_budget else None

        self._seed = params.get("seed")
        self._elite_best = float(params.get("elite_best", 0.1))
        self._elite_worst = float(params.get("elite_worst", 0.3))

        self.last_best = dict(_EMPTY_RESULT)

    def solve(self, instance: ProblemInput) -> dict:
        self.last_best = dict(_EMPTY_RESULT)

        n_aisles = len(instance.aisles)
        order_sizes = [sum(o.values()) for o in instance.orders]

        # A função fitness agora avalia corredores em vez de pedidos
        fitness = self._make_fitness(instance, order_sizes, n_aisles)

        problem_def = {
            "obj_func": fitness,
            "bounds": BinaryVar(
                n_vars=n_aisles
            ),  # O CROMOSSOMO AGORA É O NÚMERO DE CORREDORES
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

        Model = _VARIANT_MAP[self._variant]
        model = Model(**model_kwargs)

        solve_kwargs = {}

        starting = None

        try:
            starting = self._build_starting_solutions(instance, n_aisles)
        except Exception as exc:
            print(f"Error building starting solutions: {exc}")
            # Continuar sem soluções iniciais, o GA ainda pode funcionar

        if starting is not None:
            solve_kwargs["starting_solutions"] = np.asarray(starting, dtype=int)
        if self._seed is not None:
            solve_kwargs["seed"] = int(self._seed)
        if self._time_budget is not None:
            solve_kwargs["termination"] = Termination(max_time=self._time_budget)

        try:
            model.solve(problem_def, **solve_kwargs)
        except Exception as exc:
            print(f"GA encountered an error: {exc}")
            return dict(_EMPTY_RESULT)

        if self.last_best.get("selected_orders"):
            return dict(self.last_best)

        return dict(_EMPTY_RESULT)

    def _make_fitness(self, instance, order_sizes, n_aisles):
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders

        def fitness(x):
            # 1. Decodifica o cromossomo
            proposed_aisles = [i for i in range(n_aisles) if x[i] > 0.5]
            if not proposed_aisles:
                return 0.0

            # 2. Agrega o estoque disponível
            available_stock = {}
            for i in proposed_aisles:
                for item, qty in aisles[i].items():
                    available_stock[item] = available_stock.get(item, 0) + qty

            # 3. Heurística Gulosa
            order_priorities = sorted(
                range(n_orders), key=lambda idx: order_sizes[idx], reverse=True
            )

            selected_orders = []
            total_volume = 0
            temp_stock = available_stock.copy()

            for idx in order_priorities:
                order = orders[idx]
                size = order_sizes[idx]

                if total_volume + size > ub:
                    continue

                can_fulfill = True
                for item, qty in order.items():
                    if temp_stock.get(item, 0) < qty:
                        can_fulfill = False
                        break

                if can_fulfill:
                    selected_orders.append(idx)
                    total_volume += size
                    for item, qty in order.items():
                        temp_stock[item] -= qty

            if not selected_orders:
                return 0.0

            # --- MELHORIA 1: PODA (PRUNING) DOS CORREDORES NÃO UTILIZADOS ---
            # Vamos contar apenas os corredores que realmente forneceram itens
            # para os pedidos selecionados, evitando punir o GA por explorar.
            real_demand = {}
            for idx in selected_orders:
                for item, qty in orders[idx].items():
                    real_demand[item] = real_demand.get(item, 0) + qty

            used_aisles = []
            current_demand = real_demand.copy()

            for i in proposed_aisles:
                aisle_was_used = False
                for item, qty in aisles[i].items():
                    if item in current_demand and current_demand[item] > 0:
                        aisle_was_used = True
                        # Abate a demanda para não contar um corredor futuro desnecessariamente
                        current_demand[item] -= min(qty, current_demand[item])

                if aisle_was_used:
                    used_aisles.append(i)

            n_used_aisles = len(used_aisles) if used_aisles else len(proposed_aisles)

            # --- MELHORIA 2: PENALIZAÇÃO SUAVE PARA RESTRIÇÃO DE LB ---
            # Em vez de retornar 0.0 (que destrói o gradiente do GA),
            # damos uma nota proporcional para que o GA saiba que está "quase lá".
            if total_volume < lb:
                # O fator de penalidade cai drasticamente quanto mais longe do lb
                penalty_factor = (total_volume / lb) ** 2
                # Multiplicamos por um peso pequeno (ex: 0.1) para garantir
                # que NUNCA vença uma solução válida
                obj = (total_volume / n_used_aisles) * penalty_factor * 0.1
                return obj

            # 5. Calcula o objetivo (Aptidão) real da solução válida
            obj = total_volume / n_used_aisles

            # Salva a melhor solução encontrada (APENAS SE ATINGIR O LB)
            if obj > self.last_best["objective"] and total_volume >= lb:
                self.last_best = {
                    "selected_orders": sorted(selected_orders),
                    "visited_aisles": list(
                        used_aisles
                    ),  # Salva os podados, são mais eficientes
                    "objective": obj,
                }
            return obj

        return fitness

    def _build_starting_solutions(self, instance: ProblemInput, n_aisles):
        if not self._seed_with_heuristics:
            return None

        if self.start == "random":
            return self.get_random_seeds(instance, n_aisles, self._pop_size)
        elif self.start == "seed_aisle":
            return self.get_seed_aisles_seeds(instance, n_aisles)

        aisle_seeds = []

        total_demand = {}

        for order in instance.orders:
            for item, qty in order.items():
                total_demand[item] = total_demand.get(item, 0) + qty

        ordered_orders = sorted(
            range(instance.nOrders),
            key=lambda idx: sum(instance.orders[idx].values()),
            reverse=True,
        )

        ordered_aisles = sorted(
            range(n_aisles),
            key=lambda idx: sum(
                min(qty, total_demand.get(item, 0))
                for item, qty in instance.aisles[idx].items()
            ),
            reverse=True,
        )

        # sum(min(qty, gap.get(item, 0)) for item, qty in a.items())

        selected_aisles = []

        is_valid_solution = False
        idx = 0

        while not is_valid_solution and idx < n_aisles:
            selected_aisles.append(ordered_aisles[idx])

            selected_orders = []
            total_volume = 0

            stock = {}

            for aisle_idx in selected_aisles:
                for item, qty in instance.aisles[aisle_idx].items():
                    stock[item] = stock.get(item, 0) + qty

            for order_idx in ordered_orders:
                order = instance.orders[order_idx]

                if total_volume + sum(order.values()) > instance.ub:
                    continue

                if all(stock.get(item, 0) >= qty for item, qty in order.items()):
                    selected_orders.append(order_idx)
                    for item, qty in order.items():
                        stock[item] -= qty

            if sum(stock.values()) >= instance.lb:
                is_valid_solution = True
            else:
                idx += 1

        if is_valid_solution:
            mask = np.zeros(n_aisles, dtype=int)
            for a in selected_aisles:
                mask[a] = 1
            aisle_seeds.append(mask)

        if len(aisle_seeds) < self._pop_size:

            aisle_seed_seeds = self.get_seed_aisles_seeds(instance, n_aisles)

            check = aisle_seeds + aisle_seed_seeds

            if len(check) > self._pop_size:
                return check[: self._pop_size]

            if len(check) < self._pop_size:

                random_seeds = self.get_random_seeds(
                    instance, n_aisles, self._pop_size - len(check)
                )

                return aisle_seeds + random_seeds.tolist()

        return aisle_seeds

    def get_seed_aisles_seeds(self, instance, n_aisles):
        aisle_seeds = []

        ordered_orders = sorted(
            range(instance.nOrders),
            key=lambda idx: sum(instance.orders[idx].values()),
            reverse=True,
        )

        for seed_aisle_idx in range(min(self._pop_size, n_aisles)):

            ordered_aisles = sorted(
                range(n_aisles),
                key=lambda idx: similarity(
                    instance.aisles[seed_aisle_idx], instance.aisles[idx]
                ),
                reverse=True,
            )

            selected_aisles = []

            is_valid_solution = False
            idx = 0

            while not is_valid_solution and idx < n_aisles:
                selected_aisles.append(ordered_aisles[idx])

                selected_orders = []
                total_volume = 0

                stock = {}

                for aisle_idx in selected_aisles:
                    for item, qty in instance.aisles[aisle_idx].items():
                        stock[item] = stock.get(item, 0) + qty

                for order_idx in ordered_orders:
                    order = instance.orders[order_idx]

                    if total_volume + sum(order.values()) > instance.ub:
                        continue

                    if all(stock.get(item, 0) >= qty for item, qty in order.items()):
                        selected_orders.append(order_idx)
                        for item, qty in order.items():
                            stock[item] -= qty

                if sum(stock.values()) >= instance.lb:
                    is_valid_solution = True
                else:
                    idx += 1

            if is_valid_solution:
                mask = np.zeros(n_aisles, dtype=int)
                for a in selected_aisles:
                    mask[a] = 1
                aisle_seeds.append(mask)
            else:
                # Se não encontrar uma solução válida, adiciona um cromossomo vazio
                aisle_seeds.append(np.zeros(n_aisles, dtype=int))

        if len(aisle_seeds) < self._pop_size:

            random_seeds = self.get_random_seeds(
                instance, n_aisles, self._pop_size - len(aisle_seeds)
            )

            return aisle_seeds + random_seeds.tolist()

        return aisle_seeds

    def get_random_seeds(self, instance, n_aisles, n_seeds=50):
        random_seeds = np.array([])

        seed_set = set()

        for _ in range(n_seeds):

            result = SimpleHeuristic(
                {
                    "greedy": "simple",
                }
            ).solve(instance)

            visited = result.get("visited_aisles", [])

            mask = np.zeros(n_aisles, dtype=int)
            for a in visited:
                mask[a] = 1

            seed_set.add(tuple(mask))

            random_seeds = (
                np.vstack([random_seeds, mask]) if random_seeds.size > 0 else mask
            )

        return random_seeds
