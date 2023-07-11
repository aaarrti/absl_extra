from __future__ import annotations

import functools
import inspect
from importlib import util
from typing import Callable, Literal, TypeVar

from absl import logging

T = TypeVar("T", bound=Callable)


def log_exception(
    func: T | None = None, logger: Callable[[str], None] = logging.error
) -> T | Callable[[T], T]:
    """Log raised exception, and argument which caused it."""

    def decorator(func2: T) -> T:
        @functools.wraps(func2)
        def wrapper(*args, **kwargs):
            func_args = inspect.signature(func2).bind(*args, **kwargs).arguments
            func_args_str = ", ".join(
                map("{0[0]} = {0[1]!r}".format, func_args.items())
            )

            try:
                return func2(*args, **kwargs)
            except Exception as ex:
                logger(
                    f"{func2.__module__}.{func2.__qualname__} with args ( {func_args_str} ) raised {ex}"
                )
                raise ex

        return wrapper

    return decorator(func) if func is not None else decorator


def setup_logging(
    *,
    log_format: str = "%(asctime)s:[%(filename)s:%(lineno)s->%(funcName)s()]:%(levelname)s: %(message)s",
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] = "DEBUG",
):
    import logging

    import absl.logging

    logging.basicConfig(
        level=logging.getLevelName(log_level),
        format=log_format,
    )

    absl.logging.set_verbosity(absl.logging.converter.ABSL_NAMES[log_level])

    if util.find_spec("tensorflow"):
        import tensorflow as tf

        tf.get_logger().setLevel(log_level)
