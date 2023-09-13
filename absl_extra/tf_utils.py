from __future__ import annotations

import functools
import logging
import platform
from typing import Callable, Type, TypeVar

import tensorflow as tf
import toolz

from absl_extra.typing_utils import ParamSpec

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
            logging.info("Not running on linux, and linux_only==True, ignoring GPU strategy check.")
            return func(*args, **kwargs)

        gpus = tf.config.list_physical_devices("GPU")
        logging.info(f"Available GPUs -> {gpus}")
        if len(gpus) == 0:
            raise RuntimeError("No GPU available.")
        return func(*args, **kwargs)

    return wrapper


def make_tpu_strategy(
    tpu: str | None = None, experimental_spmd_xla_partitioning: bool = True
) -> tf.distribute.Strategy:
    """
    Used for testing locally scripts, which them must run on Colab TPUs. Allows to keep the same scripts,
    without changing strategy assignment.
    If running on linux, will try to create TPUStrategy. Otherwise, will return NoOpStrategy.

    Parameters
    ----------

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
        return tf.distribute.get_strategy()

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu, experimental_spmd_xla_partitioning=experimental_spmd_xla_partitioning)
    return strategy


def make_gpu_strategy(
    strategy_cls: Type[tf.distribute.Strategy] | None = None, force: bool = False, **kwargs
) -> tf.distribute.Strategy:
    """
    Useful for testing locally scripts, which must run on multiple GPUs, without changing scripts structure.

    Parameters
    ----------
    strategy_cls:
        Optional class of the strategy to use. Can be used to choose between e.g.,
        MirroredStrategy and CentralStorage strategies.
    force:

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
    gpus = tf.config.list_physical_devices("GPU")
    n_gpus = len(gpus)
    if n_gpus == 0:
        logging.warning("No GPUs found, falling back to NoOpStrategy.")
        return tf.distribute.get_strategy()
    if n_gpus == 1:
        if force:
            return tf.distribute.OneDeviceStrategy(gpus[0])
        else:
            return tf.distribute.get_strategy()

    if strategy_cls is None:
        strategy_cls = tf.distribute.MirroredStrategy

    return strategy_cls(**kwargs)


def supports_mixed_precision() -> bool:
    """Check if mixed precision is supported by available GPUs."""
    tpus = tf.config.list_logical_devices("TPU")
    if len(tpus) != 0:
        logging.info("Mixed precision OK. You should use mixed_bfloat16 for TPU.")
        return True
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        return False
    gpu_details_list = [tf.config.experimental.get_device_details(g) for g in gpus]
    for details in gpu_details_list:
        cc = details.get("compute_capability")
        if cc is None:
            return False
        if cc >= (7, 0):
            logging.info("Mixed precision OK. You should use mixed_float16 for GPU.")
            return True
    return False
