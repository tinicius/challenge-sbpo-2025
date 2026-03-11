from models.solver import Solver


class AisleCentric(Solver):

    def solve(self) -> tuple[list[int], list[int]]:
        """
            Aisle-Centric Greedy Approach (Optimized).
        """
        aisle_volumes = []
        for i, aisle in enumerate(self.aisles):
            vol = sum(aisle.values())
            aisle_volumes.append((vol, i))
        
        aisle_volumes.sort(key=lambda x: x[0], reverse=True)
        sorted_aisles = [idx for vol, idx in aisle_volumes]

        best_solution = ([], [])
        best_obj = -1.0

        order_sizes = [sum(order.values()) for order in self.orders]
        # Pre-convert orders to list of (item_id, qty) for faster iteration
        order_items = [[(item, qty) for item, qty in order.items()] for order in self.orders]

        def can_fulfill(items, current_demand, current_supply):
            # Check if current_supply can fulfill (current_demand + order)
            for item_id, qty in items:
                needed = current_demand.get(item_id, 0) + qty
                if current_supply.get(item_id, 0) < needed:
                    return False
            return True

        for start_aisle in sorted_aisles[:1]:
            selected_orders = []
            visited_aisles = {start_aisle}
            total_units = 0
            pending_orders = set(range(self.n_orders))
            current_demand = {}
            current_supply = {}
            
            for item, stock in self.aisles[start_aisle].items():
                current_supply[item] = stock

            while True:
                added_any = False
                for order_idx in list(pending_orders):
                    order_units = order_sizes[order_idx]
                    if total_units + order_units > self.ub:
                        continue
                    
                    items = order_items[order_idx]
                    if can_fulfill(items, current_demand, current_supply):
                        selected_orders.append(order_idx)
                        for item_id, qty in items:
                            current_demand[item_id] = current_demand.get(item_id, 0) + qty
                        total_units += order_units
                        pending_orders.remove(order_idx)
                        added_any = True

                if total_units >= self.lb:
                    obj = total_units / max(1, len(visited_aisles))
                    if obj > best_obj:
                        best_obj = obj
                        best_solution = (list(selected_orders), sorted(list(visited_aisles)))

                if added_any:
                    continue

                best_next_aisle = -1
                max_orders_completed = -1

                for cand_aisle in range(self.n_aisles):
                    if cand_aisle in visited_aisles:
                        continue
                    
                    # Compute temporary supply
                    temp_supply = current_supply.copy()
                    for item, stock in self.aisles[cand_aisle].items():
                        temp_supply[item] = temp_supply.get(item, 0) + stock
                        
                    completed_count = 0
                    for order_idx in pending_orders:
                        if total_units + order_sizes[order_idx] > self.ub:
                            continue
                        if can_fulfill(order_items[order_idx], current_demand, temp_supply):
                            completed_count += 1
                    
                    if completed_count > max_orders_completed:
                        max_orders_completed = completed_count
                        best_next_aisle = cand_aisle
                
                if best_next_aisle != -1 and max_orders_completed > 0:
                    visited_aisles.add(best_next_aisle)
                    for item, stock in self.aisles[best_next_aisle].items():
                        current_supply[item] = current_supply.get(item, 0) + stock
                else:
                    break

        return best_solution
