from __future__ import annotations
import functools
import logging
from typing import (
    Callable,
    TypeVar,
    Protocol,
    ContextManager,
    runtime_checkable,
    Type,
    Mapping,
)
from contextlib import contextmanager
import tensorflow as tf
import platform


R = TypeVar("R")


def requires_gpu(
    func: Callable[[...], R], linux_only: bool = False
) -> Callable[[...], R]:
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
    def wrapper(*args, **kwargs) -> R:
        if linux_only and platform.system() != "linux":
            return func(*args, **kwargs)

        if len(tf.config.list_logical_devices("GPU")) == 0:
            raise RuntimeError("No GPU available.")

        return func(*args, **kwargs)

    return wrapper


@runtime_checkable
class StrategyLike(Protocol):
    def scope(self) -> ContextManager:
        ...


class NoOpStrategy:
    def __init__(self, **kwargs):
        pass

    @contextmanager
    def scope(self) -> ContextManager:
        yield


def make_tpu_strategy(
    *,
    cluster_resolver_kwargs: Mapping[str, ...] | None = None,
    connector_kwargs: Mapping[str, ...] | None = None,
    strategy_kwargs: Mapping[str, ...] | None = None,
) -> StrategyLike:
    """
    Used for testing locally scripts, which them must run on Colab TPUs. Allows to keep the same scripts,
    without changing strategy assignment.
    If running on linux, will try to create TPUStrategy. Otherwise, will return NoOpStrategy.

    Parameters
    ----------
    cluster_resolver_kwargs:
        Kwargs passed to TPUClusterResolver.
    connector_kwargs:
        Kwargs passed to experimental_connect_to_cluster.
    strategy_kwargs:
        Kwargs passed to TPUStrategy.

    Returns
    -------

    strategy: TPUStrategy on Linux, NoOpStrategy for other OS hosts.


    Examples
    -------
    >>> strategy = make_tpu_strategy()
    >>> with strategy.scope():
    >>>     model = make_model(...)
    >>>     model.fit(...)
    """
    if platform.system().lower() != "linux":
        logging.warning("Not running on linux, falling back to NoOpStrategy.")
        return NoOpStrategy()

    if cluster_resolver_kwargs is None:
        cluster_resolver_kwargs = {}
    if connector_kwargs is None:
        connector_kwargs = {}
    if strategy_kwargs is None:
        strategy_kwargs = {}

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(**cluster_resolver_kwargs)
    tf.config.experimental_connect_to_cluster(tpu, **connector_kwargs)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu, **strategy_kwargs)
    return strategy


def make_gpu_strategy(
    strategy_cls: Type[StrategyLike] | None = None, **kwargs
) -> StrategyLike:
    """
    Useful for testing locally scripts, which must run on multiple GPUs, without changing scripts structure.

    Parameters
    ----------
    strategy_cls:
        Optional class of the strategy to use. Can be used to choose between e.g., MirroredStrategy and CentralStorage strategies.
    kwargs:
        Kwargs passed to strategy class __init__ method.


    Returns
    -------

    strategy:
        StrategyLike object.

    Examples
    -------
    >>> strategy = make_gpu_strategy()
    >>> with strategy.scope():
    >>>     model = make_model(...)
    >>>     model.fit(...)
    """
    n_gpus = len(tf.config.list_physical_devices("GPU"))
    if n_gpus == 0:
        logging.warning("No GPUs found, falling back to NoOpStrategy.")
        return NoOpStrategy()
    if n_gpus == 1:
        return tf.distribute.OneDeviceStrategy(**kwargs)

    if strategy_cls is None:
        strategy_cls = tf.distribute.MirroredStrategy

    return strategy_cls(**kwargs)


def supports_mixed_precision() -> bool:
    """Check if mixed precision is supported by available GPUs."""
    tpus = tf.config.list_logical_devices("TPU")
    if len(tpus) != 0:
        logging.info("Mixed precision OK. You should use mixed_bfloat16 for TPU.")
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        return False
    gpu_details_list = [tf.config.experimental.get_device_details(g) for g in gpus]
    for details in gpu_details_list:
        cc = details.get("compute_capability")
        if cc >= (7, 0):
            logging.info("Mixed precision OK. You should use mixed_float16 for GPU.")
            return True
    return False
