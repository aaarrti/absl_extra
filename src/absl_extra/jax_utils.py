from __future__ import annotations

import collections
import itertools
import logging
from typing import Deque, Generator, Iterable, TypeVar

import jax
from absl_extra.typing_utils import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def prefetch_to_device(
    iterator: Iterable[T], size: int = 2
) -> Generator[T, None, None]:
    """
    Parameters
    ----------
    iterator: Iterable[T]
        The input iterator to prefetch elements from.
    size: int, optional
        The number of elements to prefetch at a time. Defaults to 2.

    Returns
    -------
    Generator[T, None, None]
        A generator that yields the prefetched elements from the iterator.

    Raises
    ------
    ValueError
        If more than one GPU device is detected.

    Notes
    -----
    This method is used to prefetch elements from an iterator to a GPU device. It checks if the device is GPU and then
    enqueues *up to* `size` elements from the iterator to a deque.
    It uses JAX's `tree_map` and `device_put` functions to move
    the elements to the GPU device. The generator yields the prefetched elements one at a time.
    """
    queue: Deque[T] = collections.deque()
    devices = jax.devices()
    if devices[0].device_kind != "gpu":
        logging.error("Prefetch must be used only with GPU")
        for i in iterator:
            yield i

    if len(devices) > 1:
        raise ValueError(
            "Prefetch must be used only with single GPU, for multi-GPU support us flax.jax_utils.prefetch_to_device."
        )

    def enqueue(n: int) -> None:
        """Enqueues *up to* `n` elements from the iterator."""
        for data in itertools.islice(iterator, n):
            queue.append(
                jax.tree_util.tree_map(lambda xs: jax.device_put(xs, devices[0]), data)
            )

    enqueue(size)  # Fill up the buffer.
    while queue:
        yield queue.popleft()
        enqueue(1)
