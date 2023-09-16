from __future__ import annotations

import json
from importlib import util
from typing import TYPE_CHECKING, Protocol

from absl import flags, logging
from toolz import dicttoolz

if util.find_spec("ml_collections"):
    from ml_collections import ConfigDict
else:
    ConfigDict = None

if TYPE_CHECKING:
    from absl_extra.notifier import BaseNotifier

    class CallbackFn(Protocol):
        def __call__(
            self,
            name: str,
            *,
            notifier: BaseNotifier,
            config: ConfigDict = None,
        ) -> None:
            ...


def log_absl_flags_callback(*args, **kwargs):
    def map_fn(v):
        # In case ml collections is installed, and config dict was parsed as ABSL flags.
        if ConfigDict is not None and isinstance(v, ConfigDict):
            return v.to_dict()
        else:
            return v

    logging.info("-" * 50)
    flags_dict = dicttoolz.valmap(
        map_fn,
        flags.FLAGS.flag_values_dict().copy(),
    )
    logging.info(f"ABSL flags: {json.dumps(flags_dict, sort_keys=True, indent=4)}")


def log_tensorflow_devices(*args, **kwargs):
    """Logs the TensorFlow devices available in the system."""
    import tensorflow as tf

    logging.info(f"TF devices = {tf.config.list_physical_devices()}")


def log_jax_devices(*args, **kwargs):
    """Logs the JAX devices available in the system."""
    import jax

    logging.info(f"JAX devices = {jax.devices()}")


def log_startup_callback(name: str, *, notifier: BaseNotifier, **kwargs):
    """Notify about on execution begin."""
    notifier.notify_task_started(name)


def log_shutdown_callback(name: str, *, notifier: BaseNotifier, **kwargs):
    """Notify on task execution end."""
    notifier.notify_task_finished(name)


DEFAULT_INIT_CALLBACKS = [
    log_absl_flags_callback,
    log_startup_callback,
]
DEFAULT_POST_CALLBACK = [
    log_shutdown_callback,
]

if util.find_spec("tensorflow"):
    DEFAULT_INIT_CALLBACKS.append(log_tensorflow_devices)
if util.find_spec("jax") and util.find_spec("jaxlib"):
    DEFAULT_INIT_CALLBACKS.append(log_jax_devices)
