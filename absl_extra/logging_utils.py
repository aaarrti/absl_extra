from __future__ import annotations

import functools
import inspect
from importlib import util
from typing import Callable, TypeVar, Literal

from absl import logging

R = TypeVar("R")
LogLevel = Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def log_before(
    func: Callable[[...], R], logger: Callable[[str], None] = logging.debug
) -> Callable[[...], R]:
    """

    Parameters
    ----------
    func
    logger

    Returns
    -------

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> R:
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        logger(
            f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


def log_after(
    func: Callable[[...], R], logger: Callable[[str], None] = logging.debug
) -> Callable[[...], R]:
    """
    Log's function's return value.

    Parameters
    ----------
    func:
        Function exit from which must be logged.
    logger:
        Logger to use, default absl.logging.debug

    Returns
    -------

    func:
        Function with the same signature.

    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> R:
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
