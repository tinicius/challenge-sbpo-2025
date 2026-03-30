def multi_greedy_aisle_select(
    demand: dict[int, int], aisles: list[dict[int, int]]
) -> list[int]:
    remaining_demand = {item: qty for item, qty in demand.items() if qty > 0}

    selected_aisles: list[int] = []
    available_aisles = set(range(len(aisles)))

    while remaining_demand and available_aisles:
        best_aisle_idx = -1
        max_score = 0

        for idx in available_aisles:
            aisle = aisles[idx]
            score = sum(
                min(remaining_demand.get(item, 0), qty) for item, qty in aisle.items()
            )
            if score > max_score:
                max_score = score
                best_aisle_idx = idx

        if max_score == 0:
            break

        selected_aisles.append(best_aisle_idx)
        available_aisles.remove(best_aisle_idx)

        best_aisle = aisles[best_aisle_idx]
        for item, qty in best_aisle.items():
            if item in remaining_demand:
                remaining_demand[item] -= qty
                if remaining_demand[item] <= 0:
                    del remaining_demand[item]

    return selected_aisles
