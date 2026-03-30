from dataclasses import dataclass


@dataclass
class ProblemInput:
    """Input representation for the wave order-picking problem."""

    nOrders: int
    nItems: int
    nAisles: int
    orders: list[dict[int, int]]  # orders[o][item] = qty
    aisles: list[dict[int, int]]  # aisles[a][item] = qty
    lb: int  # wave size lower bound
    ub: int  # wave size upper bound


def load_instance(filename: str) -> ProblemInput:
    """Parse a .txt instance file into a ProblemInput."""
    with open(filename, "r") as file:
        data = file.read()

    lines = data.splitlines()

    first = lines[0].split(" ")
    nOrders = int(first[0])
    nItems = int(first[1])
    nAisles = int(first[2])

    orders = []
    for o_i in range(nOrders):
        order_line = lines[o_i + 1].split(" ")
        n_order_items = int(order_line[0])
        details = {}
        for k in range(n_order_items):
            item_idx = int(order_line[2 * k + 1])
            item_quantity = int(order_line[2 * k + 2])
            details[item_idx] = item_quantity
        orders.append(details)

    aisles = []
    for o_i in range(nAisles):
        aisle_line = lines[o_i + 1 + nOrders].split(" ")
        n_aisle_items = int(aisle_line[0])
        details = {}
        for k in range(n_aisle_items):
            item_idx = int(aisle_line[2 * k + 1])
            item_quantity = int(aisle_line[2 * k + 2])
            details[item_idx] = item_quantity
        aisles.append(details)

    wave_size_line = lines[nOrders + nAisles + 1].split(" ")
    lb = int(wave_size_line[0])
    ub = int(wave_size_line[1])

    return ProblemInput(
        nOrders=nOrders,
        nItems=nItems,
        nAisles=nAisles,
        orders=orders,
        aisles=aisles,
        lb=lb,
        ub=ub,
    )


class InstanceCache:
    """Lazy-loading cache for problem instances."""

    def __init__(self):
        self._cache: dict[str, ProblemInput] = {}

    def get(self, path: str) -> ProblemInput:
        if path not in self._cache:
            self._cache[path] = load_instance(path)
        return self._cache[path]

    def preload(self, paths: list[str]) -> None:
        for path in paths:
            self.get(path)
