from __future__ import annotations

import collections
import functools
import itertools
import logging
import platform
import sys
from typing import Callable, Deque, Generator, Iterable, TypeVar

import jax
import toolz

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


@toolz.curry
def requires_gpu(func: Callable[P, T], linux_only: bool = False) -> Callable[P, T]:
    """
    Fail if function is executing on host without access to GPU(s).
    Useful for early detecting container runtime misconfigurations.

    Parameters
    ----------
    func:
        Function, which needs hardware acceleration.
    linux_only:
        If set to true, will ignore check on non-linux hosts.


    Returns
    -------

    func:
        Function with the same signature as original one.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if linux_only and platform.system().lower() != "linux":
            logging.info(
                "Not running on linux, and linux_only==True, ignoring GPU strategy check."
            )
            return func(*args, **kwargs)

        devices = jax.devices()
        logging.info(f"JAX devices -> {devices}")
        if devices[0].device_kind != "gpu":
            raise RuntimeError("No GPU available.")
        return func(*args, **kwargs)

    return wrapper


def prefetch_to_device(
    iterator: Iterable[T], size: int = 2
) -> Generator[T, None, None]:
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
