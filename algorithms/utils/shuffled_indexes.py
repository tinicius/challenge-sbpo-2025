import random


def shuffled_indexes(x: int) -> list[int]:
    indexes = list(range(x))
    random.shuffle(indexes)
    return indexes
