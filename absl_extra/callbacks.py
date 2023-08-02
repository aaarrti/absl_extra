from __future__ import annotations

import json
from importlib import util
from typing import TYPE_CHECKING, Protocol

from absl import flags, logging

if util.find_spec("ml_collections"):
    from ml_collections import ConfigDict
else:
    ConfigDict = None
if util.find_spec("pymongo"):
    from pymongo.collection import Collection
else:
    Collection = None

if TYPE_CHECKING:
    from absl_extra.notifier import BaseNotifier


class CallbackFn(Protocol):
    def __call__(
        self,
        name: str,
        *,
        notifier: BaseNotifier,
        config: ConfigDict = None,
        db: Collection = None,
    ) -> None:
        ...


def log_absl_flags_callback(*args, **kwargs):
    logging.info("-" * 50)
    flags_dict = flags.FLAGS.flag_values_dict()
    if "config" in flags_dict:
        flags_dict["config"] = flags_dict["config"].to_dict()
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


def setup_logging(*args, **kwargs):
    log_level = flags.FLAGS.log_level
    from absl_extra.logging_utils import setup_logging

    setup_logging(log_level=log_level)  # noqa


DEFAULT_INIT_CALLBACKS = [
    setup_logging,
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
