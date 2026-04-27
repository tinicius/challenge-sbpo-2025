"""Preprocessing: inverse maps + aisle dominance pruning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from problems.base import ProblemInput


@dataclass
class Preprocessed:
    order_units: np.ndarray            # shape (n_orders,)
    order_to_items: list[set[int]]     # order_to_items[o] = {items in order o}
    aisle_to_items: list[set[int]]     # aisle_to_items[a] = {items in aisle a}
    item_to_orders: list[list[int]]    # item_to_orders[i] = [orders that ask item i]
    item_to_aisles: list[list[int]]    # item_to_aisles[i] = [aisles that have item i]
    active_aisles: list[int]           # aisles surviving dominance pruning
    total_supply: dict[int, int]       # item -> total stock across active aisles


def remove_dominated_aisles(instance: ProblemInput) -> list[int]:
    """An aisle a is dominated by a' iff items(a) ⊆ items(a') and u_{a'i} ≥ u_{ai}
    for all i ∈ items(a), with strict superiority somewhere (extra item or larger qty).
    Returns surviving aisle indices.
    """
    n = instance.nAisles
    aisles = instance.aisles
    keep = [True] * n
    item_sets = [set(a.keys()) for a in aisles]

    for a in range(n):
        if not keep[a]:
            continue
        items_a = item_sets[a]
        if not items_a:
            keep[a] = False
            continue
        a_dict = aisles[a]
        for b in range(n):
            if a == b or not keep[b]:
                continue
            items_b = item_sets[b]
            if not items_a.issubset(items_b):
                continue
            b_dict = aisles[b]
            dominates = True
            strict = len(items_b) > len(items_a)  # b has extra items
            for item in items_a:
                bq, aq = b_dict[item], a_dict[item]
                if bq < aq:
                    dominates = False
                    break
                if bq > aq:
                    strict = True
            if dominates and strict:
                keep[a] = False
                break

    return [a for a in range(n) if keep[a]]


def preprocess(instance: ProblemInput, prune_aisles: bool = True) -> Preprocessed:
    n_orders = instance.nOrders
    n_aisles = instance.nAisles
    n_items = instance.nItems

    order_units = np.array(
        [sum(o.values()) for o in instance.orders], dtype=np.int64
    )
    order_to_items = [set(o.keys()) for o in instance.orders]
    aisle_to_items = [set(a.keys()) for a in instance.aisles]

    item_to_orders: list[list[int]] = [[] for _ in range(n_items)]
    for o in range(n_orders):
        for item in order_to_items[o]:
            item_to_orders[item].append(o)

    item_to_aisles: list[list[int]] = [[] for _ in range(n_items)]
    for a in range(n_aisles):
        for item in aisle_to_items[a]:
            item_to_aisles[item].append(a)

    if prune_aisles:
        active_aisles = remove_dominated_aisles(instance)
    else:
        active_aisles = list(range(n_aisles))

    total_supply: dict[int, int] = {}
    for a in active_aisles:
        for item, qty in instance.aisles[a].items():
            total_supply[item] = total_supply.get(item, 0) + qty

    return Preprocessed(
        order_units=order_units,
        order_to_items=order_to_items,
        aisle_to_items=aisle_to_items,
        item_to_orders=item_to_orders,
        item_to_aisles=item_to_aisles,
        active_aisles=active_aisles,
        total_supply=total_supply,
    )
