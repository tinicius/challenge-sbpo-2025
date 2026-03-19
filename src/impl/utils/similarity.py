def similarity(first_order: dict[int, int], second_order: dict[int, int]) -> float:

    set1 = set(first_order.keys())
    set2 = set(second_order.keys())

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:
        return 0.0

    return len(intersection) / len(union)
