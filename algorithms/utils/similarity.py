def similarity(
    a: dict[int, int], b: dict[int, int], weighted: bool = False
) -> float:
    """Jaccard similarity between two orders.

    - weighted=False: set Jaccard over item keys, ignoring quantities.
        |keys(a) ∩ keys(b)| / |keys(a) ∪ keys(b)|
    - weighted=True: weighted Jaccard over quantities.
        Σ min(a[i], b[i]) / Σ max(a[i], b[i])  for i in keys(a) ∪ keys(b)

    Symmetric, value in [0, 1]. Returns 0.0 when the union is empty.
    """
    if not weighted:
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        union = keys_a | keys_b
        if not union:
            return 0.0
        return len(keys_a & keys_b) / len(union)

    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    num = 0
    den = 0
    for item in keys:
        qa = a.get(item, 0)
        qb = b.get(item, 0)
        num += min(qa, qb)
        den += max(qa, qb)
    if den == 0:
        return 0.0
    return num / den
