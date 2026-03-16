from abc import ABC, abstractmethod
from copy import deepcopy


class ProblemInput:
    def __init__(
        self,
        nOrders: int,
        nItems: int,
        nAisles: int,
        orders: list[dict[int, int]],
        aisles: list[dict[int, int]],
        lb: int,
        ub: int,
    ):
        self.nOrders = nOrders
        self.nItems = nItems
        self.nAisles = nAisles
        self.orders = orders
        self.aisles = aisles
        self.lb = lb
        self.ub = ub


class Solver(ABC):

    def __init__(
        self,
        input: ProblemInput,
        config: dict = {},
    ):

        input_copy = deepcopy(input)

        self.n_orders = input_copy.nOrders
        self.n_aisles = input_copy.nAisles
        self.orders = input_copy.orders
        self.aisles = input_copy.aisles
        self.lb = input_copy.lb
        self.ub = input_copy.ub
        self.config = config

    @abstractmethod
    def solve(self) -> tuple[list[int], list[int]]:
        pass
