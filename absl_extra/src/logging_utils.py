from __future__ import annotations

import functools
import inspect
from importlib import util
from typing import Callable, TypeVar, Literal, ParamSpecArgs, ParamSpecKwargs

from absl import logging

A = ParamSpecArgs("A")
K = ParamSpecKwargs("K")
R = TypeVar("R")
LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]
Func = Callable[[A, K], R]
Logger = Callable[[str], None]


def log_exception(func: Func, logger: Logger = logging.error) -> Func:
    """Log raised exception, and argument which caused it."""

    @functools.wraps(func)
    def wrapper(*args: A, **kwargs: K) -> R:
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


def log_before(func: Func, logger: Logger = logging.debug) -> Func:
    """Log argument and function name."""

    @functools.wraps(func)
    def wrapper(*args: A, **kwargs: K) -> R:
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        logger(
            f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


def log_after(func: Func, logger: Logger = logging.debug) -> Func:
    """Log's function's return value."""

    @functools.wraps(func)
    def wrapper(*args: A, **kwargs: K) -> R:
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
    log_level: LogLevel = "DEBUG",
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
