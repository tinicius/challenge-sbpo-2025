def repair_infeasible(
    orders: list[dict[int, int]],
    order_sizes: list[int],
    selected_orders: list[int],
    aisles: list[dict[int, int]],
    lb: int,
) -> list[int]:
    """
    Remove orders until combined demand is satisfiable from the full warehouse
    (total demand per item ≤ total warehouse supply per item).
    Returns reduced list, or [] if lb can no longer be reached.
    """
    total_supply: dict[int, int] = {}
    for aisle in aisles:
        for item, qty in aisle.items():
            total_supply[item] = total_supply.get(item, 0) + qty

    current = list(selected_orders)

    while current:
        demand: dict[int, int] = {}
        for o in current:
            for item, qty in orders[o].items():
                demand[item] = demand.get(item, 0) + qty

        violated = {item for item, qty in demand.items() if qty > total_supply.get(item, 0)}
        if not violated:
            break

        # Remove the order that contributes most to violated items
        to_remove = max(
            current,
            key=lambda o: sum(orders[o].get(item, 0) for item in violated),
        )
        current = [o for o in current if o != to_remove]

    if sum(order_sizes[o] for o in current) < lb:
        return []
    return current


def knapsack_dp(
    weights: list[int],
    values: list[float],
    ub: int,
    lb: int,
) -> list[int] | None:
    """
    General 0/1 Knapsack DP with integer weights and float values.

    Returns indices of items achieving max total value with total weight
    in [lb, ub], or None if no feasible subset exists.

    Falls back to greedy (value/weight ratio descending) when n * ub > 5M.
    """
    n = len(weights)
    if n == 0:
        return None

    MAX_CELLS = 5_000_000
    if n * ub > MAX_CELLS:
        order = sorted(
            range(n),
            key=lambda i: -(values[i] / weights[i]) if weights[i] > 0 else float("-inf"),
        )
        selected, total = [], 0
        for i in order:
            if total + weights[i] <= ub:
                selected.append(i)
                total += weights[i]
        return selected if total >= lb else None

    NEG_INF = float("-inf")
    # dp[w] = max total value achievable with total weight exactly w
    dp = [NEG_INF] * (ub + 1)
    dp[0] = 0.0
    keep = [[False] * (ub + 1) for _ in range(n)]

    for i, (w_i, v_i) in enumerate(zip(weights, values)):
        for w in range(ub, w_i - 1, -1):
            if dp[w - w_i] != NEG_INF and dp[w - w_i] + v_i > dp[w]:
                dp[w] = dp[w - w_i] + v_i
                keep[i][w] = True

    best_w = max(
        (w for w in range(lb, ub + 1) if dp[w] != NEG_INF),
        key=lambda w: dp[w],
        default=None,
    )
    if best_w is None:
        return None

    selected, w = [], best_w
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected.append(i)
            w -= weights[i]
    return selected
