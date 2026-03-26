def greedy_aisle_select(
    demand: dict[int, int], aisles: list[dict[int, int]]
) -> list[int]:
    def aisle_score(aisle: dict[int, int]) -> int:
        return sum(min(qty, aisle.get(item, 0)) for item, qty in demand.items())

    sorted_aisles = sorted(
        range(len(aisles)),
        key=lambda idx: aisle_score(aisles[idx]),
        reverse=True,  # best aisles first
    )

    selected_aisles: list[int] = []

    for aisle_idx in sorted_aisles:
        aisle = aisles[aisle_idx]

        if aisle_score(aisle) == 0:
            continue

        selected_aisles.append(aisle_idx)  # append once per aisle

        for item, needed in demand.items():
            if aisle.get(item, 0) > 0:
                demand[item] = max(0, needed - aisle[item])

        if sum(demand.values()) == 0:
            break

    return selected_aisles
