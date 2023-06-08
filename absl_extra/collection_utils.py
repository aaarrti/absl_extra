from __future__ import annotations

from typing import TypeVar, Mapping, Dict, Hashable, Callable

T = TypeVar("T")
R = TypeVar("R")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable, covariant=True)


def map_dict(
    dictionary: Mapping[K, T],
    *,
    value_mapper: Callable[[T], R] | None = None,
    key_mapper: Callable[[K], R] | None = None,
) -> Dict[K | R, V | R]:
    """
    Applies func to values in dict.
    Kinda like tf.nest.map_structure or jax.tree_util.tree_map, but preserves keys.
    Additionally, if provided can also map keys.
    """

    def identity(x: T) -> T:
        return x

    if value_mapper is None:
        value_mapper = identity

    if key_mapper is None:
        key_mapper = identity

    result = {}
    for k, v in dictionary.items():
        new_key = key_mapper(k)
        if isinstance(v, Mapping):
            new_value = map_dict(
                dictionary[k], value_mapper=value_mapper, key_mapper=key_mapper
            )
        else:
            new_value = value_mapper(v)
        result[new_key] = new_value
    return result


def filter_dict(
    dictionary: Mapping[K, V],
    *,
    key_filter: Callable[[K], bool] | None = None,
    value_filter: Callable[[V], bool] | None = None,
) -> Dict[K, V]:
    """
    Filters dictionary based on key or values.
    Kinda like jax.tree_util.tree_filter, but preserves keys.
    """

    def tautology(_) -> bool:
        # Tautology is an expression, which is always true.
        return True

    if key_filter is None:
        key_filter = tautology

    if value_filter is None:
        value_filter = tautology

    result = {}

    for k, v in dictionary.items():
        if key_filter(k) and value_filter(v):
            result[k] = v

    return result
