import random

from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput
from problems.validation import is_solution_feasible


class OrderOverlapCluster(Algorithm):

    def __init__(self, params: dict):
        super().__init__(params)

    @property
    def name(self) -> str:
        return "order_overlap_cluster"

    def _order_units(self, order_idx: int) -> int:
        return sum(self.orders[order_idx].values())

    def _build_order_aisle_sets(self, order_indices: list[int]) -> dict[int, set[int]]:
        item_to_aisles: dict[int, set[int]] = {}
        for aisle_idx, aisle in enumerate(self.aisles):
            for item, qty in aisle.items():
                if qty <= 0:
                    continue
                item_to_aisles.setdefault(item, set()).add(aisle_idx)

        order_aisles: dict[int, set[int]] = {}
        for order_idx in order_indices:
            covered_aisles: set[int] = set()
            for item, qty in self.orders[order_idx].items():
                if qty <= 0:
                    continue
                covered_aisles.update(item_to_aisles.get(item, set()))
            order_aisles[order_idx] = covered_aisles

        return order_aisles

    def _pairwise_overlap(
        self,
        order_indices: list[int],
        order_aisles: dict[int, set[int]],
    ) -> dict[tuple[int, int], float]:
        scores: dict[tuple[int, int], float] = {}

        for i, order_i in enumerate(order_indices):
            aisles_i = order_aisles[order_i]
            for order_j in order_indices[i + 1 :]:
                aisles_j = order_aisles[order_j]
                overlap = float(len(aisles_i & aisles_j))
                scores[(min(order_i, order_j), max(order_i, order_j))] = overlap

        return scores

    def _cluster_similarity(
        self,
        cluster_a: set[int],
        cluster_b: set[int],
        pair_scores: dict[tuple[int, int], float],
    ) -> float:
        total = 0.0
        pairs = 0

        for order_a in cluster_a:
            for order_b in cluster_b:
                key = (min(order_a, order_b), max(order_a, order_b))
                total += pair_scores.get(key, 0.0)
                pairs += 1

        if pairs == 0:
            return 0.0

        return total / pairs

    def _agglomerative_clusters(
        self,
        order_indices: list[int],
        order_units: dict[int, int],
        pair_scores: dict[tuple[int, int], float],
        target_clusters: int,
        threshold: float,
        rng: random.Random,
    ) -> list[list[int]]:
        clusters: list[set[int]] = [{order_idx} for order_idx in order_indices]
        cluster_units: list[int] = [order_units[o] for o in order_indices]

        if target_clusters < 1:
            target_clusters = 1

        while len(clusters) > target_clusters and len(clusters) > 1:
            best_pairs: list[tuple[int, int]] = []
            best_score = -1.0

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    if cluster_units[i] + cluster_units[j] > self.ub:
                        continue

                    score = self._cluster_similarity(
                        clusters[i], clusters[j], pair_scores
                    )

                    if score > best_score:
                        best_score = score
                        best_pairs = [(i, j)]
                    elif score == best_score:
                        best_pairs.append((i, j))

            if not best_pairs or best_score < threshold:
                break

            merge_i, merge_j = rng.choice(best_pairs)
            clusters[merge_i] = clusters[merge_i] | clusters[merge_j]
            cluster_units[merge_i] = cluster_units[merge_i] + cluster_units[merge_j]

            del clusters[merge_j]
            del cluster_units[merge_j]

        return [sorted(list(cluster)) for cluster in clusters if cluster]

    def _build_solution_for_orders(
        self, selected_orders: list[int]
    ) -> tuple[list[int], float]:
        if not selected_orders:
            return [], 0.0

        demand, total_units = self._aggregate_demand(selected_orders)

        if total_units < self.lb or total_units > self.ub:
            repaired = self._repair_infeasible_by_removing_orders(selected_orders)
            if repaired is None:
                return [], 0.0
            repaired_orders, repaired_aisles = repaired
            repaired_units = sum(self.order_units[o] for o in repaired_orders)
            return repaired_aisles, repaired_units / len(repaired_aisles)

        visited_aisles = self._select_aisles_for_demand(demand)

        if not visited_aisles:
            repaired = self._repair_infeasible_by_removing_orders(selected_orders)
            if repaired is None:
                return [], 0.0
            repaired_orders, repaired_aisles = repaired
            repaired_units = sum(self.order_units[o] for o in repaired_orders)
            return repaired_aisles, repaired_units / len(repaired_aisles)

        if not is_solution_feasible(self.inst, selected_orders, visited_aisles):
            repaired = self._repair_infeasible_by_removing_orders(selected_orders)
            if repaired is None:
                return [], 0.0
            repaired_orders, repaired_aisles = repaired
            repaired_units = sum(self.order_units[o] for o in repaired_orders)
            return repaired_aisles, repaired_units / len(repaired_aisles)

        return visited_aisles, total_units / len(visited_aisles)

    def _aggregate_demand(
        self, selected_orders: list[int]
    ) -> tuple[dict[int, int], int]:
        demand: dict[int, int] = {}
        total_units = 0

        for order_idx in selected_orders:
            order = self.orders[order_idx]
            total_units += sum(order.values())
            for item, qty in order.items():
                demand[item] = demand.get(item, 0) + qty

        return demand, total_units

    def _select_aisles_for_demand(self, demand: dict[int, int]) -> list[int]:
        if not demand:
            return []

        selector = (
            multi_greedy_aisle_select
            if self.greedy_mode == "multi"
            else greedy_aisle_select
        )
        return selector(dict(demand), self.aisles)

    def _infeasibility_gap(
        self, selected_orders: list[int], visited_aisles: list[int]
    ) -> int:
        demand, _ = self._aggregate_demand(selected_orders)
        gap = 0

        for item, required in demand.items():
            available = sum(self.aisles[a].get(item, 0) for a in visited_aisles)
            if required > available:
                gap += required - available

        return gap

    def _repair_infeasible_by_removing_orders(
        self, selected_orders: list[int]
    ) -> tuple[list[int], list[int]] | None:
        current_orders = sorted(set(selected_orders))

        while current_orders:
            _, current_units = self._aggregate_demand(current_orders)

            if current_units < self.lb:
                return None

            if current_units <= self.ub:
                demand, _ = self._aggregate_demand(current_orders)
                current_aisles = self._select_aisles_for_demand(demand)
                if current_aisles and is_solution_feasible(
                    self.inst, current_orders, current_aisles
                ):
                    return current_orders, current_aisles

            best_next_orders: list[int] | None = None
            best_key: tuple[float, float, float] | None = None

            for removed_order in current_orders:
                candidate_orders = [o for o in current_orders if o != removed_order]
                if not candidate_orders:
                    continue

                _, candidate_units = self._aggregate_demand(candidate_orders)
                if candidate_units < self.lb:
                    continue

                demand, _ = self._aggregate_demand(candidate_orders)
                candidate_aisles = self._select_aisles_for_demand(demand)

                if not candidate_aisles:
                    key = (float("-inf"), float("-inf"), -float(candidate_units))
                else:
                    feasible = is_solution_feasible(
                        self.inst, candidate_orders, candidate_aisles
                    )
                    gap = self._infeasibility_gap(candidate_orders, candidate_aisles)
                    objective = (
                        candidate_units / len(candidate_aisles)
                        if feasible
                        else -float(gap)
                    )
                    # Prefer lower gap first, then better objective, then larger kept units.
                    key = (-float(gap), objective, float(candidate_units))

                if best_key is None or key > best_key:
                    best_key = key
                    best_next_orders = candidate_orders

            if best_next_orders is None:
                return None

            current_orders = best_next_orders

        return None

    def _score_clusters(self, clusters: list[list[int]]) -> list[dict]:
        scored: list[dict] = []

        for cluster_orders in clusters:
            repaired = self._repair_infeasible_by_removing_orders(cluster_orders)
            if repaired is None:
                continue

            repaired_orders, aisles = repaired
            total_units = sum(self.order_units[o] for o in repaired_orders)
            if total_units <= 0 or total_units > self.ub:
                continue

            if total_units < self.lb:
                continue

            if not aisles:
                continue

            objective = total_units / len(aisles)

            scored.append(
                {
                    "orders": repaired_orders,
                    "units": total_units,
                    "aisles": aisles,
                    "objective": objective,
                }
            )

        return scored

    def _pack_clusters(self, scored_clusters: list[dict], rng: random.Random) -> dict:
        if not scored_clusters:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        shuffled = scored_clusters[:]
        rng.shuffle(shuffled)
        ordered = sorted(
            shuffled,
            key=lambda c: (c["objective"], c["units"], -len(c["aisles"])),
            reverse=True,
        )

        selected_orders: list[int] = []
        selected_set: set[int] = set()
        best_partial = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        for cluster in ordered:
            overlap = any(order_idx in selected_set for order_idx in cluster["orders"])
            if overlap:
                continue

            candidate_orders = selected_orders + cluster["orders"]
            if sum(self.order_units[o] for o in candidate_orders) > self.ub:
                continue

            candidate_aisles, candidate_obj = self._build_solution_for_orders(
                candidate_orders
            )
            if not candidate_aisles:
                continue

            selected_orders = candidate_orders
            selected_set = set(selected_orders)

            candidate_units = sum(self.order_units[o] for o in selected_orders)
            if candidate_units >= self.lb and candidate_obj > best_partial["objective"]:
                best_partial = {
                    "selected_orders": sorted(selected_orders),
                    "visited_aisles": candidate_aisles,
                    "objective": candidate_obj,
                }

        if best_partial["selected_orders"]:
            return best_partial

        for cluster in ordered:
            if cluster["units"] >= self.lb:
                return {
                    "selected_orders": sorted(cluster["orders"]),
                    "visited_aisles": cluster["aisles"],
                    "objective": cluster["objective"],
                }

        return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

    def solve(self, instance: ProblemInput) -> dict:
        self.inst = self._prepare_instance(instance)
        self.orders = self.inst.orders
        self.aisles = self.inst.aisles
        self.lb = self.inst.lb
        self.ub = self.inst.ub

        if self.inst.nOrders == 0 or self.inst.nAisles == 0:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        self.greedy_mode = self.params.get("greedy", "multi")
        n_clusters_list = self.params.get("n_clusters_list", [3, 5, 10, 20])
        threshold = float(self.params.get("similarity_threshold", 0.0))
        restarts = int(self.params.get("restarts", 3))

        self.order_units = {
            order_idx: self._order_units(order_idx)
            for order_idx in range(self.inst.nOrders)
        }

        candidate_orders = [
            order_idx
            for order_idx in range(self.inst.nOrders)
            if 0 < self.order_units[order_idx] <= self.ub
        ]

        if not candidate_orders:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        order_aisles = self._build_order_aisle_sets(candidate_orders)
        pair_scores = self._pairwise_overlap(candidate_orders, order_aisles)

        normalized_targets: list[int] = []
        for n_clusters in n_clusters_list:
            n = int(n_clusters)
            if n < 1:
                n = 1
            if n > len(candidate_orders):
                n = len(candidate_orders)
            if n not in normalized_targets:
                normalized_targets.append(n)

        if len(candidate_orders) not in normalized_targets:
            normalized_targets.append(len(candidate_orders))

        best = {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        for target_clusters in normalized_targets:
            for restart_idx in range(max(1, restarts)):
                rng = random.Random((target_clusters + 1) * 1009 + restart_idx * 9176)

                clusters = self._agglomerative_clusters(
                    candidate_orders,
                    self.order_units,
                    pair_scores,
                    target_clusters,
                    threshold,
                    rng,
                )

                scored_clusters = self._score_clusters(clusters)
                candidate = self._pack_clusters(scored_clusters, rng)

                if candidate["objective"] > best["objective"]:
                    best = candidate

        if best["selected_orders"]:
            repaired = self._repair_infeasible_by_removing_orders(
                best["selected_orders"]
            )
            if repaired is None:
                return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

            repaired_orders, repaired_aisles = repaired
            repaired_units = sum(self.order_units[o] for o in repaired_orders)
            return {
                "selected_orders": repaired_orders,
                "visited_aisles": repaired_aisles,
                "objective": repaired_units / len(repaired_aisles),
            }

        return best
