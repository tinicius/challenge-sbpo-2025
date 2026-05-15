from algorithms.base import Algorithm
from algorithms.utils.greedy_aisle_select import greedy_aisle_select
from algorithms.utils.multi_greedy_aisle_select import multi_greedy_aisle_select
from problems.base import ProblemInput


class SavingsHeuristic(Algorithm):
    """
    Clarke and Wright (1964) adapted Savings Heuristic for order picking.
    Supports both Absolute Savings and Normalized Savings (Bozer and Kile, 2008).
    Adapted for single-wave picking by forming all possible batches and 
    selecting the best one that satisfies the capacity constraints.
    """

    def __init__(self, params: dict):
        super().__init__(params)
        self._normalized = params.get("normalized", True)
        self._greedy = params.get("greedy", "multi")
        
        if self._greedy not in {"simple", "multi"}:
            raise ValueError(f"SavingsHeuristic: invalid 'greedy'={self._greedy!r}")

    @property
    def name(self) -> str:
        return "savings_heuristic"

    def solve(self, instance: ProblemInput) -> dict:
        orders = instance.orders
        aisles = instance.aisles
        lb, ub = instance.lb, instance.ub
        n_orders = instance.nOrders
        n_aisles = instance.nAisles

        if n_orders == 0 or n_aisles == 0:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        order_sizes = [sum(o.values()) for o in orders]
        stock = self._aggregate_stock(aisles)
        footprints = self._build_footprints(orders, aisles)

        # 1. Compute pairwise savings
        savings = []
        for i in range(n_orders):
            if order_sizes[i] == 0 or order_sizes[i] > ub:
                continue
            
            # Check if order i can be satisfied by itself
            if not all(stock.get(it, 0) >= q for it, q in orders[i].items()):
                continue

            for j in range(i + 1, n_orders):
                if order_sizes[j] == 0 or order_sizes[j] > ub:
                    continue
                
                # Check combined size constraint early
                if order_sizes[i] + order_sizes[j] > ub:
                    continue
                    
                intersection = len(footprints[i] & footprints[j])
                
                if self._normalized:
                    union_or_sum = len(footprints[i]) + len(footprints[j])
                    s = intersection / union_or_sum if union_or_sum > 0 else 0.0
                else:
                    s = intersection
                
                # Use combined size as a secondary tie-breaker to favor larger batches
                savings.append((s, order_sizes[i] + order_sizes[j], i, j))

        # 2. Sort savings descending
        # Sort primarily by saving value, secondarily by combined size
        savings.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # 3. Merge batches
        batch_id = {i: i for i in range(n_orders)}
        batch_orders = {i: [i] for i in range(n_orders)}
        batch_sizes = {i: order_sizes[i] for i in range(n_orders)}
        
        # Track stock for each batch
        batch_stock_usage = {i: dict(orders[i]) for i in range(n_orders)}
        
        # Remove invalid orders from batches
        for i in range(n_orders):
            if order_sizes[i] == 0 or order_sizes[i] > ub or not all(stock.get(it, 0) >= q for it, q in orders[i].items()):
                if i in batch_id:
                    del batch_id[i]
                    del batch_orders[i]
                    del batch_sizes[i]
                    del batch_stock_usage[i]

        for s, _, i, j in savings:
            if i not in batch_id or j not in batch_id:
                continue
            
            b_i = batch_id[i]
            b_j = batch_id[j]

            if b_i != b_j:
                new_size = batch_sizes[b_i] + batch_sizes[b_j]
                if new_size <= ub:
                    # Check stock constraints
                    stock_ok = True
                    for it, q in batch_stock_usage[b_j].items():
                        if batch_stock_usage[b_i].get(it, 0) + q > stock.get(it, 0):
                            stock_ok = False
                            break
                    
                    if stock_ok:
                        # Merge b_j into b_i
                        for order_idx in batch_orders[b_j]:
                            batch_id[order_idx] = b_i
                            batch_orders[b_i].append(order_idx)
                        
                        batch_sizes[b_i] = new_size
                        
                        for it, q in batch_stock_usage[b_j].items():
                            batch_stock_usage[b_i][it] = batch_stock_usage[b_i].get(it, 0) + q
                        
                        del batch_orders[b_j]
                        del batch_sizes[b_j]
                        del batch_stock_usage[b_j]

        # 4. Evaluate all valid batches
        best_batch = None
        best_obj = -1.0
        best_visited = []

        for b_id, b_orders in batch_orders.items():
            b_size = batch_sizes[b_id]
            if b_size >= lb and b_size <= ub:
                demand = batch_stock_usage[b_id]
                
                visited = (
                    multi_greedy_aisle_select(demand, aisles)
                    if self._greedy == "multi"
                    else greedy_aisle_select(demand, aisles)
                )
                
                if visited:
                    obj = b_size / len(visited)
                    if obj > best_obj:
                        best_obj = obj
                        best_batch = list(b_orders)
                        best_visited = visited

        if not best_batch:
            return {"selected_orders": [], "visited_aisles": [], "objective": 0.0}

        return {
            "selected_orders": best_batch,
            "visited_aisles": best_visited,
            "objective": best_obj,
        }

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
    def _aggregate_stock(aisles: list[dict[int, int]]) -> dict[int, int]:
        stock: dict[int, int] = {}
        for aisle in aisles:
            for item, qty in aisle.items():
                stock[item] = stock.get(item, 0) + qty
        return stock
