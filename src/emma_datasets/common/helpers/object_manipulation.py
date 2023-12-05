import itertools
from typing import TypeVar


S = TypeVar("S")
T = TypeVar("T")


def flip_list_map_elements(previous_map: dict[S, list[T]]) -> dict[T, list[S]]:
    """Flip a mapping of elements to a list by using the list elements as keys."""
    all_value_elements = set(itertools.chain.from_iterable(previous_map.values()))

    newly_mapped_elements: dict[T, list[S]] = {element: [] for element in all_value_elements}

    for previous_key, previous_value_list in previous_map.items():
        for new_key in previous_value_list:
            newly_mapped_elements[new_key].append(previous_key)

    return newly_mapped_elements
