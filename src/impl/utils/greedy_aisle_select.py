def greedy_aisle_select(
    demand: dict[int, int], aisles: list[dict[int, int]]
) -> list[int]:
    # 1. Prevent side-effects by creating a local copy of unfulfilled demand
    remaining_demand = {item: qty for item, qty in demand.items() if qty > 0}

    selected_aisles: list[int] = []
    available_aisles = set(range(len(aisles)))

    # Loop dynamically until demand is met or we run out of useful aisles
    while remaining_demand and available_aisles:
        best_aisle_idx = -1
        max_score = 0

        # 2. Dynamic scoring: Find the aisle that satisfies the MOST REMAINING demand
        for idx in available_aisles:
            aisle = aisles[idx]

            # 3. Efficiency: Iterate through the aisle's items, not the whole demand
            score = sum(
                min(remaining_demand.get(item, 0), qty) for item, qty in aisle.items()
            )

            if score > max_score:
                max_score = score
                best_aisle_idx = idx

        # If the best aisle provides 0 items we need, no other aisle can help us
        if max_score == 0:
            break

        selected_aisles.append(best_aisle_idx)
        available_aisles.remove(best_aisle_idx)

        # 4. Update demand and remove items that hit 0 so the dictionary shrinks
        best_aisle = aisles[best_aisle_idx]
        for item, qty in best_aisle.items():
            if item in remaining_demand:
                remaining_demand[item] -= qty
                if remaining_demand[item] <= 0:
                    del remaining_demand[item]

    return selected_aisles
