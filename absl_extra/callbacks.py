from __future__ import annotations

from typing import Protocol, TYPE_CHECKING
from importlib import util
from absl import logging, flags
import json

if util.find_spec("ml_collections"):
    from ml_collections import ConfigDict
else:
    ConfigDict = None
if util.find_spec("pymongo"):
    from pymongo.collection import Collection
else:
    Collection = None

if TYPE_CHECKING:
    from absl_extra.src.notifier import BaseNotifier


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


def log_params_callback(name: str, *, config: ConfigDict = None, **kwargs):
    logging.info("-" * 50)
    logging.info(
        f"Flags: {json.dumps(flags.FLAGS.flag_values_dict(), sort_keys=True, indent=4)}"
    )

    if config is not None:
        logging.info(
            f"Config: {json.dumps(config.to_dict(), sort_keys=True, indent=4)}"
        )

    logging.info("-" * 50)


def log_startup_callback(name: str, *, notifier: BaseNotifier, **kwargs):
    notifier.notify_task_started(name)


def log_shutdown_callback(name: str, *, notifier: BaseNotifier, **kwargs):
    notifier.notify_task_finished(name)
