from abc import ABC, abstractmethod


class Solver(ABC):

    def __init__(self, n_orders: int, n_aisles: int, orders: list[dict], aisles: list[dict], lb: int, ub: int):
        self.n_orders = n_orders
        self.n_aisles = n_aisles
        self.orders = orders
        self.aisles = aisles
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def solve(self) -> tuple[list[int], list[int]]:
        pass
