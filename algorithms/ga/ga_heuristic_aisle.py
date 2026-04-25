import numpy as np
from mealpy import BinaryVar, Termination
from mealpy.evolutionary_based import GA

from algorithms.base import Algorithm
from algorithms.simple.simple_heuristic import SimpleHeuristic
from algorithms.aisle_first.aisle_first_heuristic import AisleFirstHeuristic
from problems.base import ProblemInput

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


class AisleBasedGeneticAlgorithm(Algorithm):
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
        starting = self._build_starting_solutions(instance, n_aisles)
        if starting is not None:
            solve_kwargs["starting_solutions"] = np.asarray(starting, dtype=int)
        if self._seed is not None:
            solve_kwargs["seed"] = int(self._seed)
        if self._time_budget is not None:
            solve_kwargs["termination"] = Termination(max_time=self._time_budget)

        model.solve(problem_def, **solve_kwargs)

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

    def _build_starting_solutions(self, instance, n_aisles):
        if not self._seed_with_heuristics:
            return None

        heur_seeds = []

        # --- MELHORIA 3: CORREÇÃO DO BUG DAS SEMENTES REPETIDAS ---
        # Definimos parâmetros variados para gerar diversidade real na população inicial
        heur_configs = [
            {"score": "useful", "order": "desc", "prune": "multi"},
            {"score": "useful", "order": "asc", "prune": "multi"},
            {"score": "density", "order": "desc", "prune": "multi"},
            {
                "score": "random",
                "order": "desc",
                "prune": "multi",
            },  # Insere um pouco de caos benéfico
        ]

        for heur_params in heur_configs:
            try:
                # Passa o heur_params dinamicamente em vez de hardcoded!
                r = AisleFirstHeuristic(heur_params).solve(instance)

                if r.get("selected_orders"):
                    demand = {}
                    for idx in r["selected_orders"]:
                        for item, qty in instance.orders[idx].items():
                            demand[item] = demand.get(item, 0) + qty

                    visited = multi_greedy_aisle_select(demand, instance.aisles)

                    mask = np.zeros(n_aisles, dtype=int)
                    for a in visited:
                        mask[a] = 1

                    # Adiciona à semente apenas se for diferente das já existentes (evita clones)
                    if not any(
                        np.array_equal(mask, existing) for existing in heur_seeds
                    ):
                        heur_seeds.append(mask)
            except Exception:
                continue

        if not heur_seeds:
            return None

        rng = np.random.default_rng(self._seed)
        rest = self._pop_size - len(heur_seeds)

        # Para os randômicos, ao invés de puro 50/50, podemos inicializar com menos corredores
        # (já que a heurística tende a podar). Ex: 30% de chance de ter o corredor = 1
        random_seeds = [
            (rng.random(n_aisles) < 0.3).astype(int) for _ in range(max(0, rest))
        ]

        return heur_seeds + random_seeds
