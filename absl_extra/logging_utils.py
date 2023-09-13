from __future__ import annotations

import functools
import inspect
from importlib import util
from traceback import format_exception
from types import FunctionType, MethodType
from typing import Callable, Literal, OrderedDict, Sequence, TypeVar

import toolz
from absl import logging

from absl_extra.typing_utils import ParamSpec

R = TypeVar("R")
P = ParamSpec("P")


@toolz.curry
def log_exception(
    func: Callable[P, R],
    logger: Callable[[str], None] = logging.error,
    ignore_argnums: Sequence[int] = (),
    ignore_argnames: Sequence[str] = (),
) -> Callable[P, R]:
    """
    Log raised exception, and argument which caused it.

    Parameters
    ----------
    func:
        Function, which is expected to raise.
    logger:
        Logging function, default absl.logging.error
    ignore_argnums:
        Positional arguments, which must not be logged.
    ignore_argnames:
        Keyword arguments, which must bot be logged.

    Returns
    -------

    func:
        Function with same signature.

    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            func_name = format_callable_name(func)
            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            func_args_str = format_callable_args(func_args, ignore_argnums, ignore_argnames)
            logger(f"{func_name} with args ( {func_args_str} ) raised {format_exception(ex)}")
            raise

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
    func: Callable[P, R],
    logger: Callable[[str], None] = logging.debug,
    ignore_argnums: Sequence[int] = (),
    ignore_argnames: Sequence[str] = (),
) -> Callable[P, R]:
    """

    Log functions argument before calling it.

    Parameters
    ----------
    func
    logger:
        Logging function, default absl.logging.debug
    ignore_argnums:
        Positional arguments, which must not be logged.
    ignore_argnames:
        Keyword arguments, which must bot be logged.

    Returns
    -------


    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = format_callable_args(func_args, ignore_argnums, ignore_argnames)
        func_name_str = format_callable_name(func)
        logger(f"Entered {func_name_str} with args ( {func_args_str} )")
        return func(*args, **kwargs)

    return wrapper


@toolz.curry
def log_after(
    func: Callable[P, R],
    logger: Callable[[str], None] = logging.debug,
    ignore_nums: Sequence[int] = (),
) -> Callable[P, R]:
    """Log's function's return value."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        retval = func(*args, **kwargs)
        func_name = format_callable_name(func)

        if isinstance(retval, tuple):
            filtered_retval = tuple([val for i, val in enumerate(retval) if i not in ignore_nums])
        else:
            filtered_retval = retval  # type: ignore

        logger(f"Exited {func_name}(...) with return value: {repr(filtered_retval)}")
        return retval

    return wrapper


def format_callable_name(func: Callable[P, R]) -> str:
    if inspect.ismethod(func):
        _method: MethodType = func
        return f"{_method.__module__}.{_method.__class__}.{_method.__qualname__}"
    else:
        _func: FunctionType = func  # type: ignore
        return f"{_func.__module__}.{_func.__qualname__}"


def format_callable_args(
    arguments: OrderedDict[str, ...],  # type: ignore
    ignore_argnums: Sequence[int] = (),
    ignore_argnames: Sequence[str] = (),
) -> str:
    filtered_args = {}

    for i, (k, v) in enumerate(arguments.items()):
        if i not in ignore_argnums and k not in ignore_argnames:
            filtered_args[k] = v

    return ", ".join(map("{0[0]} = {0[1]!r}".format, filtered_args.items()))
