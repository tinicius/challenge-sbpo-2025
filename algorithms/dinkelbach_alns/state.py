"""WaveSolution: incremental state for the Dinkelbach-ALNS metaheuristic.

The state tracks the current order/aisle selection plus per-item demand and
coverage so that operators can evaluate moves in O(items_in_order) instead of
recomputing from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from problems.base import ProblemInput


@dataclass
class WaveSolution:
    orders: set[int] = field(default_factory=set)
    aisles: set[int] = field(default_factory=set)
    total_units: int = 0
    item_demand: dict[int, int] = field(default_factory=dict)
    item_covered: dict[int, int] = field(default_factory=dict)

    def ratio(self) -> float:
        if not self.aisles:
            return 0.0
        return self.total_units / len(self.aisles)

    def dinkelbach_value(self, lam: float) -> float:
        return self.total_units - lam * len(self.aisles)

    def is_feasible(self, instance: ProblemInput) -> bool:
        if not (instance.lb <= self.total_units <= instance.ub):
            return False
        for item, demand in self.item_demand.items():
            if self.item_covered.get(item, 0) < demand:
                return False
        return True

    def copy(self) -> WaveSolution:
        return WaveSolution(
            orders=self.orders.copy(),
            aisles=self.aisles.copy(),
            total_units=self.total_units,
            item_demand=self.item_demand.copy(),
            item_covered=self.item_covered.copy(),
        )

    @classmethod
    def from_sets(
        cls,
        orders: set[int] | list[int],
        aisles: set[int] | list[int],
        instance: ProblemInput,
    ) -> WaveSolution:
        orders_set = set(orders)
        aisles_set = set(aisles)

        item_demand: dict[int, int] = {}
        total_units = 0
        for o in orders_set:
            for item, qty in instance.orders[o].items():
                item_demand[item] = item_demand.get(item, 0) + qty
                total_units += qty

        # item_covered tracks coverage for ALL items in current aisles (not just
        # demanded ones). This keeps the invariant valid when future orders
        # introduce new demanded items already supplied by existing aisles.
        item_covered: dict[int, int] = {}
        for a in aisles_set:
            for item, qty in instance.aisles[a].items():
                item_covered[item] = item_covered.get(item, 0) + qty

        return cls(
            orders=orders_set,
            aisles=aisles_set,
            total_units=total_units,
            item_demand=item_demand,
            item_covered=item_covered,
        )
