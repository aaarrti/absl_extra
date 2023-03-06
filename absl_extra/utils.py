from __future__ import annotations

from typing import Callable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def value_or_default(value: T, default_factory: Callable[[], T]) -> T:
    if value is not None:
        return value

    return default_factory()


def map_optional(value: Optional[T], fn: Callable[[T], R]) -> Optional[R]:
    if value is None:
        return None
    return fn(value)
