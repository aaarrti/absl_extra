from __future__ import annotations

import functools
import inspect
from importlib import util
from typing import Callable, Literal, TypeVar

import toolz
from absl import logging
from absl_extra.typing_utils import ParamSpec


T = TypeVar("T")
P = ParamSpec("P")


@toolz.curry
def log_exception(
    func: Callable[P, T], logger: Callable[[str], None] = logging.error
) -> Callable[P, T]:
    """Log raised exception, and argument which caused it."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
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


@toolz.curry
def log_before(
    func: Callable[P, T], logger: Callable[[str], None] = logging.debug
) -> Callable[P, T]:
    """Log functions argument before calling it."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        logger(
            f"Entered {func.__module__}.{_get_func_name(func)} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


@toolz.curry
def log_after(
    func: Callable[P, T], logger: Callable[[str], None] = logging.debug
) -> Callable[P, T]:
    """Log's function's return value."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        retval = func(*args, **kwargs)
        logger(
            f"Exited {func.__module__}.{func.__qualname__}(...) with value: "
            + repr(retval)
        )
        return retval

    return wrapper


def _get_func_name(func):
    if isinstance(func, functools.partial):
        func = func.func
    return func.__qualname__
