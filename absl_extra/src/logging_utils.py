from __future__ import annotations

import functools
import inspect
from importlib import util
from typing import Callable, Literal, TypeVar

from absl import logging

C = TypeVar("C", bound=Callable)


def log_exception(func: C, logger: Callable[[str], None] = logging.error) -> C:
    """Log raised exception, and argument which caused it."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))

        try:
            return func(*args, **kwargs)
        except Exception as ex:
            logger(
                f"{func.__module__}.{func.__qualname__} with args ( {func_args_str} ) raised {ex}"
            )
            raise ex

    return wrapper


def log_before(func: C, logger: Callable[[str], None] = logging.debug) -> C:
    """Log argument and function name."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        logger(
            f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


def log_after(func, logger: Callable[[str], None] = logging.debug):
    """Log's function's return value."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retval = func(*args, **kwargs)
        logger(
            f"Exited {func.__module__}.{func.__qualname__}(...) with value: "
            + repr(retval)
        )
        return retval

    return wrapper


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
