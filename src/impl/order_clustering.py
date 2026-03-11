from models.solver import Solver


class OrderClustering(Solver):

    def solve(self) -> tuple[list[int], list[int]]:
        """
            Order-Clustering (Seed Algorithm) for the Wave Order Picking problem.

            1. Pick a "seed" order.
            2. Calculate "similarity score" (Jaccard) to all other orders based on required aisles.
            3. Group high-similarity orders until wave size is within [lb, ub].
            4. Select the best wave among all seeds.
        """

        best_solution = ([], [])
        best_obj = -1.0

        # Precompute possible aisles for each item
        item_to_aisles = {}
        for aisle_idx, aisle in enumerate(self.aisles):
            for item, stock in aisle.items():
                if stock > 0:
                    item_to_aisles.setdefault(item, set()).add(aisle_idx)

        # Compute Footprint for each order
        order_footprints = []
        for order in self.orders:
            footprint = set()
            for item in order:
                if item in item_to_aisles:
                    footprint.update(item_to_aisles[item])
            order_footprints.append(footprint)

        def jaccard(set1, set2):
            if not set1 and not set2: return 0.0
            return len(set1 & set2) / len(set1 | set2)

        order_sizes = [sum(order.values()) for order in self.orders]
        
        # Select up to 20 largest orders as seeds
        seed_candidates = sorted(range(self.n_orders), key=lambda x: order_sizes[x], reverse=True)[:20]

        for seed_idx in seed_candidates:
            seed_order = self.orders[seed_idx]
            seed_footprint = order_footprints[seed_idx]
            seed_size = sum(seed_order.values())

            if seed_size > self.ub:
                continue

            wave = [seed_idx]
            wave_size = seed_size

            # Compute similarities
            similarities = []
            for other_idx in range(self.n_orders):
                if other_idx == seed_idx:
                    continue
                sim = jaccard(seed_footprint, order_footprints[other_idx])
                similarities.append((sim, other_idx))

            # Sort by highest similarity
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Greedily form the cluster
            for sim, other_idx in similarities:
                other_size = sum(self.orders[other_idx].values())
                if wave_size + other_size <= self.ub:
                    wave.append(other_idx)
                    wave_size += other_size

            if wave_size >= self.lb:
                # Calculate minimum necessary aisles for this wave
                total_demand = {}
                for order_idx in wave:
                    for item, qty in self.orders[order_idx].items():
                        total_demand[item] = total_demand.get(item, 0) + qty

                visited_aisles = set()
                current_supply = {}

                for item_id, needed in sorted(total_demand.items()):
                    if current_supply.get(item_id, 0) >= needed:
                        continue

                    # Need more supply for this item — iterate aisles
                    for aisle_idx in range(self.n_aisles):
                        if aisle_idx in visited_aisles:
                            continue
                        stock = self.aisles[aisle_idx].get(item_id, 0)
                        if stock > 0:
                            visited_aisles.add(aisle_idx)
                            # Update current supply with all items from this new aisle
                            for itm, stk in self.aisles[aisle_idx].items():
                                current_supply[itm] = current_supply.get(itm, 0) + stk
                                
                            if current_supply.get(item_id, 0) >= needed:
                                break

                # Verify feasibility (all items can be supplied)
                is_feasible = True
                for item_id, needed in total_demand.items():
                    if current_supply.get(item_id, 0) < needed:
                        is_feasible = False
                        break

                if is_feasible:
                    obj = wave_size / len(visited_aisles)
                    if obj > best_obj:
                        best_obj = obj
                        best_solution = (list(wave), sorted(list(visited_aisles)))

        return best_solution
