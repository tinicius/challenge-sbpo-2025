from abc import ABC, abstractmethod
from copy import deepcopy

from problems.base import ProblemInput


class Algorithm(ABC):
    """Abstract base class that every algorithm must implement."""

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def solve(self, instance: ProblemInput) -> dict:
        """
        Receives a problem instance.
        Returns a result dict with at minimum:
          - 'selected_orders': list[int]
          - 'visited_aisles': list[int]
          - 'objective': float
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this algorithm."""
        ...

    def _prepare_instance(self, instance: ProblemInput) -> ProblemInput:
        """Deep-copy the instance to ensure solver isolation."""
        return deepcopy(instance)
